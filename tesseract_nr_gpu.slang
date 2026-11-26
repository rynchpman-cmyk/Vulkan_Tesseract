// GPU-resident 1D Theory 3.3 Carter evolution.
//
// State columns:
//   0 D_N, 1 D_D, 2 S_x, 3 E, 4 P_x^D, 5 D_N*s, 6 D_N*Y.
//
// Recovery variables:
//   0 log(n), 1 q_N, 2 log(d), 3 q_D, 4 specific entropy.
//
// The first production milestone intentionally uses the analytic M1 closure,
// a causal local-Lax-Friedrichs flux, the literal carrier path product, and
// flat one-dimensional spacetime. All stage physics stays device resident.

import "slangpy";
import "theory33_qualified_frozen.slang";

static const int VT_STATE_WIDTH = 7;
static const int VT_RECOVERY_WIDTH = 5;
static const int VT_DIAGNOSTIC_WIDTH = 14;
static const int VT_PATH_WIDTH = 4;
static const int VT_EIGEN_AUDIT_WIDTH = 8;

static const float VT_GAMMA_AD = 1.6666666666666667;
static const float VT_CARRIER_K = 0.2;
static const float VT_CARRIER_SOUND_SQUARED = 0.81;
static const float VT_CHEMICAL_SUSCEPTIBILITY = 10.0;
static const float VT_TARGET_FRACTION = 0.05;
static const float VT_DENSITY_FLOOR = 1.0e-8;
static const float VT_CARRIER_FLOOR = 1.0e-8;
static const float VT_RECOVERY_TOLERANCE = 3.0e-5;
static const float VT_EPSILON = 1.0e-12;
static const float VT_CHARACTERISTIC_STEP9 = 1.0e-3;

struct VTConstitutive
{
    float lambdaValue;
    float bN;
    float bD;
    float entrainment;
    float generalizedPressure;
    float legendreMinimum;
    float thermodynamicMinimum;
    bool phaseTwo;
    float phaseThreshold;
    float switchingMargin;
};

struct VTPrediction
{
    float conserved[5];
    float flux[5];
    float chiZero;
    float chiDown;
    float carrierEulerian;
    float carrierTransported;
    float relativeGamma;
    float velocityN;
    float velocityD;
    float entropy;
    float legendreMinimum;
    float thermodynamicMinimum;
    bool phaseTwo;
    float phaseThreshold;
    float switchingMargin;
};

struct VTRecovery
{
    float x[VT_RECOVERY_WIDTH];
    VTPrediction prediction;
    float residual;
    bool valid;
    int phaseCrossings;
};

float vtMinimumEigenvalue2(float diagonal0, float offDiagonal, float diagonal1)
{
    float difference = diagonal0 - diagonal1;
    return 0.5 * (
        diagonal0
        + diagonal1
        - sqrt(max(
            difference * difference + 4.0 * offDiagonal * offDiagonal,
            0.0
        ))
    );
}

float vtQualifiedDriftDerivative(float y)
{
    return
        0.892004527 * y
        + 0.00117659738 * (5.0 * qualifiedSigmoid(5.0 * y) - 2.5)
        + 0.00125993402 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.111111111)) - 1.8228822)
        + 0.00141513187 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.222222222)) - 1.23831901)
        + 0.00172678169 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.333333333)) - 0.794345524)
        + 0.00243270284 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.444444444)) - 0.488862999)
        + 0.00427972126 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.555555556)) - 0.292684514)
        + 0.00926524641 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.666666667)) - 0.172225978)
        + 0.0197211307 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.777777778)) - 0.100287683)
        + 0.0348923822 * (5.0 * qualifiedSigmoid(5.0 * (y - 0.888888889)) - 0.0580365822)
        + 0.0517620529 * (5.0 * qualifiedSigmoid(5.0 * (y - 1.0)) - 0.0334642546);
}

VTConstitutive vtConstitutive(
    float n,
    float d,
    float cross,
    float entropy,
    int modelMode,
    int phaseOverride
)
{
    float pressure =
        exp((VT_GAMMA_AD - 1.0) * entropy) * pow(n, VT_GAMMA_AD);
    float thermal = n + pressure / (VT_GAMMA_AD - 1.0);
    float anchor = d - VT_TARGET_FRACTION * n;
    float carrierEnergy =
        VT_CARRIER_K * pow(d, 1.0 + VT_CARRIER_SOUND_SQUARED);
    float chemicalEnergy =
        0.5 * anchor * anchor / VT_CHEMICAL_SUSCEPTIBILITY;
    float rhoN =
        1.0
        + VT_GAMMA_AD
        * pressure
        / max((VT_GAMMA_AD - 1.0) * n, VT_EPSILON)
        - VT_TARGET_FRACTION
        * anchor
        / VT_CHEMICAL_SUSCEPTIBILITY;
    float rhoD =
        VT_CARRIER_K
        * (1.0 + VT_CARRIER_SOUND_SQUARED)
        * pow(d, VT_CARRIER_SOUND_SQUARED)
        + anchor / VT_CHEMICAL_SUSCEPTIBILITY;
    float lambdaValue = -(thermal + carrierEnergy + chemicalEnergy);
    float gradientN = -rhoN / max(2.0 * n, VT_EPSILON);
    float gradientD = -rhoD / max(2.0 * d, VT_EPSILON);
    float gradientCross = 0.0;
    bool phaseTwo = false;
    float phaseThreshold = 1.0e30;
    float switchingMargin = -1.0e30;

    if (modelMode == 1)
    {
        float relativeGamma = cross / max(n * d, VT_EPSILON);
        float drift = max(relativeGamma - 1.0, 0.0);
        phaseThreshold = qualifiedPhaseThreshold(n, d, entropy);
        switchingMargin = drift - phaseThreshold;
        bool classifiedPhase = switchingMargin >= 0.0;
        phaseTwo = phaseOverride >= 0
            ? phaseOverride != 0
            : classifiedPhase;
        if (phaseTwo)
        {
            float y = drift / T33Q_DRIFT_SCALE;
            float thresholdY = phaseThreshold / T33Q_DRIFT_SCALE;
            lambdaValue -= T33Q_ENERGY_SCALE * (
                qualifiedDriftNetwork(y)
                - qualifiedDriftNetwork(thresholdY)
            );

            float logTemperatureSpan =
                log(T33Q_TEMPERATURE_MAX)
                - log(T33Q_TEMPERATURE_MIN);
            float fractionSpan =
                T33Q_FRACTION_MAX - T33Q_FRACTION_MIN;
            float fraction = d / max(n, VT_EPSILON);
            float thresholdN =
                T33Q_PHASE_N
                * 2.0
                / (T33Q_DENSITY_MAX - T33Q_DENSITY_MIN)
                + T33Q_PHASE_T
                * 2.0
                * (VT_GAMMA_AD - 1.0)
                / max(n * logTemperatureSpan, VT_EPSILON)
                - T33Q_PHASE_F
                * 2.0
                * fraction
                / max(n * fractionSpan, VT_EPSILON);
            float thresholdD =
                T33Q_PHASE_F
                * 2.0
                / max(n * fractionSpan, VT_EPSILON);
            float thresholdEntropy =
                T33Q_PHASE_T
                * 2.0
                * (VT_GAMMA_AD - 1.0)
                / logTemperatureSpan;
            float driftDerivative = vtQualifiedDriftDerivative(y);
            float thresholdDerivative =
                vtQualifiedDriftDerivative(thresholdY);
            float driftGradientN =
                -relativeGamma
                / max(2.0 * n * n * T33Q_DRIFT_SCALE, VT_EPSILON);
            float driftGradientD =
                -relativeGamma
                / max(2.0 * d * d * T33Q_DRIFT_SCALE, VT_EPSILON);
            float driftGradientCross =
                1.0 / max(n * d * T33Q_DRIFT_SCALE, VT_EPSILON);
            float thresholdGradientN =
                thresholdN
                / max(2.0 * n * T33Q_DRIFT_SCALE, VT_EPSILON);
            float thresholdGradientD =
                thresholdD
                / max(2.0 * d * T33Q_DRIFT_SCALE, VT_EPSILON);
            float thresholdGradientEntropy =
                thresholdEntropy / T33Q_DRIFT_SCALE;
            gradientN -= T33Q_ENERGY_SCALE * (
                driftDerivative * driftGradientN
                - thresholdDerivative * thresholdGradientN
            );
            gradientD -= T33Q_ENERGY_SCALE * (
                driftDerivative * driftGradientD
                - thresholdDerivative * thresholdGradientD
            );
            gradientCross -= T33Q_ENERGY_SCALE * (
                driftDerivative * driftGradientCross
            );
            // The entropy component is part of the qualified invariant
            // gradient even though the present material slice does not
            // transport its conjugate as an independent Carter field.
            float gradientEntropy =
                -pressure
                + T33Q_ENERGY_SCALE
                * thresholdDerivative
                * thresholdGradientEntropy;
        }
    }

    float bN = -2.0 * gradientN;
    float bD = -2.0 * gradientD;
    float entrainment = -gradientCross;
    float generalizedPressure =
        lambdaValue
        + bN * n * n
        + bD * d * d
        + 2.0 * entrainment * cross;
    float hessianNN =
        VT_GAMMA_AD * pressure / max(n * n, VT_EPSILON)
        + VT_TARGET_FRACTION
        * VT_TARGET_FRACTION
        / VT_CHEMICAL_SUSCEPTIBILITY;
    float hessianND =
        -VT_TARGET_FRACTION / VT_CHEMICAL_SUSCEPTIBILITY;
    float hessianDD =
        VT_CARRIER_K
        * (1.0 + VT_CARRIER_SOUND_SQUARED)
        * VT_CARRIER_SOUND_SQUARED
        * pow(d, VT_CARRIER_SOUND_SQUARED - 1.0)
        + 1.0 / VT_CHEMICAL_SUSCEPTIBILITY;

    VTConstitutive result;
    result.lambdaValue = lambdaValue;
    result.bN = bN;
    result.bD = bD;
    result.entrainment = entrainment;
    result.generalizedPressure = generalizedPressure;
    result.legendreMinimum = vtMinimumEigenvalue2(
        bN, entrainment, bD
    );
    result.thermodynamicMinimum = vtMinimumEigenvalue2(
        hessianNN, hessianND, hessianDD
    );
    result.phaseTwo = phaseTwo;
    result.phaseThreshold = phaseThreshold;
    result.switchingMargin = switchingMargin;
    return result;
}

VTPrediction vtPredict(
    in float x[VT_RECOVERY_WIDTH],
    int modelMode,
    int phaseOverride
)
{
    float n = exp(clamp(x[0], -20.0, 20.0));
    float qN = x[1];
    float d = exp(clamp(x[2], -20.0, 20.0));
    float qD = x[3];
    float entropy = clamp(x[4], -20.0, 20.0);
    float wN = sqrt(1.0 + qN * qN);
    float wD = sqrt(1.0 + qD * qD);
    float velocityN = qN / wN;
    float velocityD = qD / wD;
    float relativeGamma = max(wN * wD - qN * qD, 1.0);
    float cross = n * d * relativeGamma;
    VTConstitutive material = vtConstitutive(
        n, d, cross, entropy, modelMode, phaseOverride
    );
    float dN = n * wN;
    float dD = d * wD;
    float momentum =
        material.bN * dN * dN * velocityN
        + material.bD * dD * dD * velocityD
        + material.entrainment
        * dN
        * dD
        * (velocityN + velocityD);
    float energy =
        -material.generalizedPressure
        + material.bN * dN * dN
        + material.bD * dD * dD
        + 2.0 * material.entrainment * dN * dD;
    float chiDown =
        material.bD * dD * velocityD
        + material.entrainment * dN * velocityN;
    float chiNormal =
        material.bD * dD + material.entrainment * dN;
    float canonicalMomentum = dD * chiDown;
    float stress =
        material.generalizedPressure
        + material.bN * dN * dN * velocityN * velocityN
        + material.bD * dD * dD * velocityD * velocityD
        + 2.0
        * material.entrainment
        * dN
        * dD
        * velocityN
        * velocityD;

    VTPrediction result;
    result.conserved[0] = dN;
    result.conserved[1] = dD;
    result.conserved[2] = momentum;
    result.conserved[3] = energy;
    result.conserved[4] = canonicalMomentum;
    result.flux[0] = dN * velocityN;
    result.flux[1] = dD * velocityD;
    result.flux[2] = stress;
    result.flux[3] = momentum;
    result.flux[4] = canonicalMomentum * velocityD;
    result.chiZero = -chiNormal;
    result.chiDown = chiDown;
    result.carrierEulerian = dD;
    result.carrierTransported = dD * velocityD;
    result.relativeGamma = relativeGamma;
    result.velocityN = velocityN;
    result.velocityD = velocityD;
    result.entropy = entropy;
    result.legendreMinimum = material.legendreMinimum;
    result.thermodynamicMinimum = material.thermodynamicMinimum;
    result.phaseTwo = material.phaseTwo;
    result.phaseThreshold = material.phaseThreshold;
    result.switchingMargin = material.switchingMargin;
    return result;
}

bool vtSolve5(
    in float matrixInput[25],
    in float rhsInput[VT_RECOVERY_WIDTH],
    out float solution[VT_RECOVERY_WIDTH]
)
{
    float matrix[25] = {};
    float rhs[VT_RECOVERY_WIDTH] = {};
    for (int index = 0; index < 25; ++index)
        matrix[index] = matrixInput[index];
    for (int index = 0; index < VT_RECOVERY_WIDTH; ++index)
    {
        rhs[index] = rhsInput[index];
        solution[index] = 0.0;
    }

    for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
    {
        int pivot = column;
        float pivotMagnitude = abs(matrix[column * 5 + column]);
        for (int row = column + 1; row < VT_RECOVERY_WIDTH; ++row)
        {
            float magnitude = abs(matrix[row * 5 + column]);
            if (magnitude > pivotMagnitude)
            {
                pivot = row;
                pivotMagnitude = magnitude;
            }
        }
        if (pivotMagnitude < 1.0e-10 || !isfinite(pivotMagnitude))
            return false;
        if (pivot != column)
        {
            for (int entry = column; entry < VT_RECOVERY_WIDTH; ++entry)
            {
                float temporary = matrix[column * 5 + entry];
                matrix[column * 5 + entry] = matrix[pivot * 5 + entry];
                matrix[pivot * 5 + entry] = temporary;
            }
            float temporaryRhs = rhs[column];
            rhs[column] = rhs[pivot];
            rhs[pivot] = temporaryRhs;
        }
        float diagonal = matrix[column * 5 + column];
        for (int entry = column; entry < VT_RECOVERY_WIDTH; ++entry)
            matrix[column * 5 + entry] /= diagonal;
        rhs[column] /= diagonal;
        for (int row = 0; row < VT_RECOVERY_WIDTH; ++row)
        {
            if (row == column)
                continue;
            float factor = matrix[row * 5 + column];
            for (int entry = column; entry < VT_RECOVERY_WIDTH; ++entry)
                matrix[row * 5 + entry] -= factor * matrix[column * 5 + entry];
            rhs[row] -= factor * rhs[column];
        }
    }
    for (int index = 0; index < VT_RECOVERY_WIDTH; ++index)
        solution[index] = rhs[index];
    return true;
}

float vtResidual(
    VTPrediction prediction,
    in float target[VT_RECOVERY_WIDTH],
    in float scale[VT_RECOVERY_WIDTH],
    out float residual[VT_RECOVERY_WIDTH]
)
{
    float norm = 0.0;
    for (int row = 0; row < VT_RECOVERY_WIDTH; ++row)
    {
        residual[row] =
            (prediction.conserved[row] - target[row]) / scale[row];
        norm = max(norm, abs(residual[row]));
    }
    return norm;
}

VTRecovery vtRecoverBranch(
    in float target[VT_RECOVERY_WIDTH],
    in float initial[VT_RECOVERY_WIDTH],
    int modelMode,
    int phaseOverride
)
{
    float scale[VT_RECOVERY_WIDTH];
    float energyScale = max(max(abs(target[2]), abs(target[3])), 1.0e-7);
    scale[0] = max(abs(target[0]), 1.0e-7);
    scale[1] = max(abs(target[1]), 1.0e-7);
    scale[2] = energyScale;
    scale[3] = energyScale;
    scale[4] = max(max(abs(target[4]), abs(target[1])), 1.0e-7);

    VTRecovery result = {};
    for (int index = 0; index < VT_RECOVERY_WIDTH; ++index)
        result.x[index] = initial[index];
    result.x[0] = clamp(result.x[0], log(VT_DENSITY_FLOOR), 4.0);
    result.x[2] = clamp(result.x[2], log(VT_CARRIER_FLOOR), 4.0);
    result.x[1] = clamp(result.x[1], -5.0, 5.0);
    result.x[3] = clamp(result.x[3], -5.0, 5.0);
    result.x[4] = clamp(result.x[4], -20.0, 20.0);

    float residual[VT_RECOVERY_WIDTH] = {};
    float norm = 1.0e30;
    [MaxIters(24)]
    for (int iteration = 0; iteration < 24; ++iteration)
    {
        result.prediction = vtPredict(
            result.x, modelMode, phaseOverride
        );
        norm = vtResidual(result.prediction, target, scale, residual);
        if (norm < VT_RECOVERY_TOLERANCE)
            break;

        float jacobian[25] = {};
        for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
        {
            float step = 3.0e-4 * max(1.0, abs(result.x[column]));
            float plusX[VT_RECOVERY_WIDTH] = {};
            float minusX[VT_RECOVERY_WIDTH] = {};
            for (int index = 0; index < VT_RECOVERY_WIDTH; ++index)
            {
                plusX[index] = result.x[index];
                minusX[index] = result.x[index];
            }
            plusX[column] += step;
            minusX[column] -= step;
            VTPrediction plusPrediction = vtPredict(
                plusX, modelMode, phaseOverride
            );
            VTPrediction minusPrediction = vtPredict(
                minusX, modelMode, phaseOverride
            );
            for (int row = 0; row < VT_RECOVERY_WIDTH; ++row)
            {
                jacobian[row * 5 + column] =
                    (
                        plusPrediction.conserved[row]
                        - minusPrediction.conserved[row]
                    )
                    / (2.0 * step * scale[row]);
            }
        }

        float rhs[VT_RECOVERY_WIDTH] = {};
        float delta[VT_RECOVERY_WIDTH] = {};
        for (int row = 0; row < VT_RECOVERY_WIDTH; ++row)
            rhs[row] = -residual[row];
        if (!vtSolve5(jacobian, rhs, delta))
            break;

        bool accepted = false;
        float factor = 1.0;
        [MaxIters(10)]
        for (int lineSearch = 0; lineSearch < 10; ++lineSearch)
        {
            float trial[VT_RECOVERY_WIDTH] = {};
            for (int index = 0; index < VT_RECOVERY_WIDTH; ++index)
                trial[index] = result.x[index] + factor * delta[index];
            trial[0] = clamp(trial[0], log(VT_DENSITY_FLOOR), 4.0);
            trial[2] = clamp(trial[2], log(VT_CARRIER_FLOOR), 4.0);
            trial[1] = clamp(trial[1], -5.0, 5.0);
            trial[3] = clamp(trial[3], -5.0, 5.0);
            trial[4] = clamp(trial[4], -20.0, 20.0);
            VTPrediction trialPrediction = vtPredict(
                trial, modelMode, phaseOverride
            );
            float trialResidual[VT_RECOVERY_WIDTH];
            float trialNorm = vtResidual(
                trialPrediction, target, scale, trialResidual
            );
            if (trialNorm < norm)
            {
                for (int index = 0; index < VT_RECOVERY_WIDTH; ++index)
                    result.x[index] = trial[index];
                accepted = true;
                break;
            }
            factor *= 0.5;
        }
        if (!accepted)
            break;
    }

    result.prediction = vtPredict(
        result.x, modelMode, phaseOverride
    );
    norm = vtResidual(result.prediction, target, scale, residual);
    result.residual = norm;
    result.valid =
        isfinite(norm)
        && norm < 5.0e-4
        && result.prediction.conserved[0] > VT_DENSITY_FLOOR
        && result.prediction.conserved[1] > VT_CARRIER_FLOOR
        && result.prediction.relativeGamma <= 2.0
        && result.prediction.legendreMinimum > 0.0
        && result.prediction.thermodynamicMinimum > 0.0;
    result.phaseCrossings = 0;
    return result;
}

VTRecovery vtRecover(
    in float target[VT_RECOVERY_WIDTH],
    in float initial[VT_RECOVERY_WIDTH],
    int modelMode
)
{
    float working[VT_RECOVERY_WIDTH] = {};
    for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
        working[column] = initial[column];
    int phaseOverride = -1;
    if (modelMode == 1)
    {
        VTPrediction initialPrediction = vtPredict(
            working, modelMode, -1
        );
        phaseOverride = initialPrediction.phaseTwo ? 1 : 0;
    }

    VTRecovery result = {};
    int crossings = 0;
    [MaxIters(3)]
    for (int transition = 0; transition < 3; ++transition)
    {
        result = vtRecoverBranch(
            target, working, modelMode, phaseOverride
        );
        if (modelMode != 1)
            break;
        int classifiedPhase =
            result.prediction.switchingMargin >= 0.0 ? 1 : 0;
        if (classifiedPhase == phaseOverride)
            break;
        phaseOverride = classifiedPhase;
        crossings += 1;
        for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
            working[column] = result.x[column];
    }
    result.phaseCrossings = crossings;
    return result;
}

struct VTEigensystem
{
    float symbol[25];
    float right[25];
    float left[25];
    float values[5];
    float maximumResidual;
    float inverseResidual;
    float conditionEstimate;
    float minimumSeparation;
    bool valid;
};

bool vtInvert5(in float matrix[25], out float inverse[25])
{
    for (int index = 0; index < 25; ++index)
        inverse[index] = 0.0;
    for (int column = 0; column < 5; ++column)
    {
        float rhs[5] = {};
        float solution[5] = {};
        rhs[column] = 1.0;
        if (!vtSolve5(matrix, rhs, solution))
            return false;
        for (int row = 0; row < 5; ++row)
            inverse[row * 5 + column] = solution[row];
    }
    return true;
}

void vtMatMul5(
    in float left[25],
    in float right[25],
    out float output[25]
)
{
    for (int row = 0; row < 5; ++row)
    {
        for (int column = 0; column < 5; ++column)
        {
            float value = 0.0;
            for (int inner = 0; inner < 5; ++inner)
            {
                value +=
                    left[row * 5 + inner]
                    * right[inner * 5 + column];
            }
            output[row * 5 + column] = value;
        }
    }
}

void vtQrEigenvalues(in float input[25], out float values[5])
{
    float matrix[25] = {};
    for (int index = 0; index < 25; ++index)
        matrix[index] = input[index];

    [MaxIters(72)]
    for (int iteration = 0; iteration < 72; ++iteration)
    {
        float shift =
            matrix[24]
            + 1.0e-3 * (1.0 + abs(matrix[24]));
        float q[25] = {};
        float r[25] = {};
        for (int column = 0; column < 5; ++column)
        {
            float work[5] = {};
            for (int row = 0; row < 5; ++row)
            {
                work[row] =
                    matrix[row * 5 + column]
                    - (row == column ? shift : 0.0);
            }
            for (int previous = 0; previous < column; ++previous)
            {
                float projection = 0.0;
                for (int row = 0; row < 5; ++row)
                    projection += q[row * 5 + previous] * work[row];
                r[previous * 5 + column] = projection;
                for (int row = 0; row < 5; ++row)
                    work[row] -= projection * q[row * 5 + previous];
            }
            float normSquared = 0.0;
            for (int row = 0; row < 5; ++row)
                normSquared += work[row] * work[row];
            float norm = sqrt(max(normSquared, 0.0));
            if (norm < 1.0e-9)
            {
                for (int row = 0; row < 5; ++row)
                    work[row] = row == column ? 1.0 : 0.0;
                for (int previous = 0; previous < column; ++previous)
                {
                    float projection = 0.0;
                    for (int row = 0; row < 5; ++row)
                        projection += q[row * 5 + previous] * work[row];
                    for (int row = 0; row < 5; ++row)
                        work[row] -= projection * q[row * 5 + previous];
                }
                normSquared = 0.0;
                for (int row = 0; row < 5; ++row)
                    normSquared += work[row] * work[row];
                norm = sqrt(max(normSquared, 1.0e-20));
            }
            r[column * 5 + column] = norm;
            for (int row = 0; row < 5; ++row)
                q[row * 5 + column] = work[row] / norm;
        }
        float next[25] = {};
        vtMatMul5(r, q, next);
        for (int index = 0; index < 5; ++index)
            next[index * 5 + index] += shift;
        for (int index = 0; index < 25; ++index)
            matrix[index] = next[index];
    }
    for (int index = 0; index < 5; ++index)
        values[index] = matrix[index * 5 + index];
    for (int first = 0; first < 5; ++first)
    {
        for (int second = first + 1; second < 5; ++second)
        {
            if (values[second] < values[first])
            {
                float temporary = values[first];
                values[first] = values[second];
                values[second] = temporary;
            }
        }
    }
}

VTEigensystem vtEigensystem(in float symbol[25])
{
    VTEigensystem result = {};
    for (int index = 0; index < 25; ++index)
        result.symbol[index] = symbol[index];
    vtQrEigenvalues(symbol, result.values);

    for (int mode = 0; mode < 5; ++mode)
    {
        float vector[5] = {};
        vector[mode] = 1.0;
        float spectralShift =
            2.0e-4 * (1.0 + abs(result.values[mode]));
        [MaxIters(16)]
        for (int iteration = 0; iteration < 16; ++iteration)
        {
            float shifted[25] = {};
            for (int index = 0; index < 25; ++index)
                shifted[index] = symbol[index];
            for (int index = 0; index < 5; ++index)
            {
                shifted[index * 5 + index] -=
                    result.values[mode] + spectralShift;
            }
            float next[5] = {};
            if (!vtSolve5(shifted, vector, next))
                break;
            float normSquared = 0.0;
            for (int row = 0; row < 5; ++row)
                normSquared += next[row] * next[row];
            float norm = sqrt(max(normSquared, 1.0e-30));
            for (int row = 0; row < 5; ++row)
                vector[row] = next[row] / norm;
        }
        int pivot = 0;
        float pivotMagnitude = abs(vector[0]);
        for (int row = 1; row < 5; ++row)
        {
            if (abs(vector[row]) > pivotMagnitude)
            {
                pivot = row;
                pivotMagnitude = abs(vector[row]);
            }
        }
        float signValue = vector[pivot] < 0.0 ? -1.0 : 1.0;
        for (int row = 0; row < 5; ++row)
            result.right[row * 5 + mode] = signValue * vector[row];
    }

    bool inverseValid = vtInvert5(result.right, result.left);
    result.maximumResidual = 0.0;
    for (int row = 0; row < 5; ++row)
    {
        for (int mode = 0; mode < 5; ++mode)
        {
            float applied = 0.0;
            for (int column = 0; column < 5; ++column)
            {
                applied +=
                    symbol[row * 5 + column]
                    * result.right[column * 5 + mode];
            }
            result.maximumResidual = max(
                result.maximumResidual,
                abs(
                    applied
                    - result.values[mode]
                    * result.right[row * 5 + mode]
                )
            );
        }
    }
    result.inverseResidual = 0.0;
    float rightNorm = 0.0;
    float leftNorm = 0.0;
    for (int row = 0; row < 5; ++row)
    {
        for (int column = 0; column < 5; ++column)
        {
            float product = 0.0;
            for (int inner = 0; inner < 5; ++inner)
            {
                product +=
                    result.left[row * 5 + inner]
                    * result.right[inner * 5 + column];
            }
            result.inverseResidual = max(
                result.inverseResidual,
                abs(product - (row == column ? 1.0 : 0.0))
            );
            rightNorm +=
                result.right[row * 5 + column]
                * result.right[row * 5 + column];
            leftNorm +=
                result.left[row * 5 + column]
                * result.left[row * 5 + column];
        }
    }
    result.conditionEstimate = sqrt(rightNorm * leftNorm);
    result.minimumSeparation = 1.0e30;
    for (int mode = 0; mode < 4; ++mode)
    {
        result.minimumSeparation = min(
            result.minimumSeparation,
            abs(result.values[mode + 1] - result.values[mode])
        );
    }
    result.valid =
        inverseValid
        && isfinite(result.maximumResidual)
        && result.maximumResidual < 5.0e-3
        && result.inverseResidual < 5.0e-3
        && result.conditionEstimate < 1.0e6;
    return result;
}

VTEigensystem vtCarterSymbol(
    in float primitive[5],
    int modelMode,
    int phaseOverride
)
{
    float timeJacobian[25] = {};
    float fluxPathJacobian[25] = {};
    VTPrediction base = vtPredict(
        primitive, modelMode, phaseOverride
    );
    for (int column = 0; column < 5; ++column)
    {
        float step = 3.0e-4 * max(1.0, abs(primitive[column]));
        float plusX[5] = {};
        float minusX[5] = {};
        for (int index = 0; index < 5; ++index)
        {
            plusX[index] = primitive[index];
            minusX[index] = primitive[index];
        }
        plusX[column] += step;
        minusX[column] -= step;
        VTPrediction plusPrediction = vtPredict(
            plusX, modelMode, phaseOverride
        );
        VTPrediction minusPrediction = vtPredict(
            minusX, modelMode, phaseOverride
        );
        float inverseStep = 1.0 / (2.0 * step);
        for (int row = 0; row < 5; ++row)
        {
            timeJacobian[row * 5 + column] =
                (
                    plusPrediction.conserved[row]
                    - minusPrediction.conserved[row]
                )
                * inverseStep;
            fluxPathJacobian[row * 5 + column] =
                (
                    plusPrediction.flux[row]
                    - minusPrediction.flux[row]
                )
                * inverseStep;
        }
        float chiZeroDerivative =
            (plusPrediction.chiZero - minusPrediction.chiZero)
            * inverseStep;
        float chiDownDerivative =
            (plusPrediction.chiDown - minusPrediction.chiDown)
            * inverseStep;
        fluxPathJacobian[4 * 5 + column] -=
            base.carrierEulerian * chiZeroDerivative
            + base.carrierTransported * chiDownDerivative;
    }
    float inverseTime[25] = {};
    float symbol[25] = {};
    if (!vtInvert5(timeJacobian, inverseTime))
        return vtEigensystem(symbol);
    vtMatMul5(fluxPathJacobian, inverseTime, symbol);
    return vtEigensystem(symbol);
}

struct VTPrediction9
{
    float conserved[9];
    float entropy;
    float relativeGamma;
    float temperature;
    bool phaseTwo;
    float switchingMargin;
};

float vtDot3(in float left[3], in float right[3])
{
    return
        left[0] * right[0]
        + left[1] * right[1]
        + left[2] * right[2];
}

VTPrediction9 vtPredict9(
    in float x[9],
    int modelMode,
    int phaseOverride
)
{
    float n = exp(clamp(x[0], -20.0, 20.0));
    float entropy = clamp(x[1], -20.0, 20.0);
    float qN[3] = {};
    qN[0] = x[2];
    qN[1] = x[3];
    qN[2] = x[4];
    float d = exp(clamp(x[5], -20.0, 20.0));
    float qD[3] = {};
    qD[0] = x[6];
    qD[1] = x[7];
    qD[2] = x[8];
    float wN = sqrt(1.0 + vtDot3(qN, qN));
    float wD = sqrt(1.0 + vtDot3(qD, qD));
    float velocityN[3] = {};
    float velocityD[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        velocityN[axis] = qN[axis] / wN;
        velocityD[axis] = qD[axis] / wD;
    }
    float relativeGamma = max(
        wN * wD - vtDot3(qN, qD), 1.0
    );
    float cross = n * d * relativeGamma;
    VTConstitutive material = vtConstitutive(
        n, d, cross, entropy, modelMode, phaseOverride
    );
    float dN = n * wN;
    float dD = d * wD;

    VTPrediction9 result = {};
    result.conserved[0] = dN;
    result.conserved[1] = dD;
    for (int axis = 0; axis < 3; ++axis)
    {
        result.conserved[2 + axis] =
            material.bN * dN * dN * velocityN[axis]
            + material.bD * dD * dD * velocityD[axis]
            + material.entrainment
            * dN
            * dD
            * (velocityN[axis] + velocityD[axis]);
    }
    result.conserved[5] =
        -material.generalizedPressure
        + material.bN * dN * dN
        + material.bD * dD * dD
        + 2.0 * material.entrainment * dN * dD;
    for (int axis = 0; axis < 3; ++axis)
    {
        float chiDown =
            material.bD * dD * velocityD[axis]
            + material.entrainment * dN * velocityN[axis];
        result.conserved[6 + axis] = dD * chiDown;
    }
    float pressure =
        exp((VT_GAMMA_AD - 1.0) * entropy)
        * pow(n, VT_GAMMA_AD);
    result.entropy = entropy;
    result.relativeGamma = relativeGamma;
    result.temperature = pressure / max(n, VT_EPSILON);
    result.phaseTwo = material.phaseTwo;
    result.switchingMargin = material.switchingMargin;
    return result;
}

bool vtSolve9(
    in float matrixInput[81],
    in float rhsInput[9],
    out float solution[9]
)
{
    float matrix[81] = {};
    float rhs[9] = {};
    for (int index = 0; index < 81; ++index)
        matrix[index] = matrixInput[index];
    for (int index = 0; index < 9; ++index)
    {
        rhs[index] = rhsInput[index];
        solution[index] = 0.0;
    }
    for (int column = 0; column < 9; ++column)
    {
        int pivot = column;
        float pivotMagnitude = abs(matrix[column * 9 + column]);
        for (int row = column + 1; row < 9; ++row)
        {
            float magnitude = abs(matrix[row * 9 + column]);
            if (magnitude > pivotMagnitude)
            {
                pivot = row;
                pivotMagnitude = magnitude;
            }
        }
        if (pivotMagnitude < 1.0e-11 || !isfinite(pivotMagnitude))
            return false;
        if (pivot != column)
        {
            for (int entry = column; entry < 9; ++entry)
            {
                float temporary = matrix[column * 9 + entry];
                matrix[column * 9 + entry] =
                    matrix[pivot * 9 + entry];
                matrix[pivot * 9 + entry] = temporary;
            }
            float temporaryRhs = rhs[column];
            rhs[column] = rhs[pivot];
            rhs[pivot] = temporaryRhs;
        }
        float diagonal = matrix[column * 9 + column];
        for (int entry = column; entry < 9; ++entry)
            matrix[column * 9 + entry] /= diagonal;
        rhs[column] /= diagonal;
        for (int row = 0; row < 9; ++row)
        {
            if (row == column)
                continue;
            float factor = matrix[row * 9 + column];
            for (int entry = column; entry < 9; ++entry)
            {
                matrix[row * 9 + entry] -=
                    factor * matrix[column * 9 + entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (int index = 0; index < 9; ++index)
        solution[index] = rhs[index];
    return true;
}

float vtDragResidual(
    VTPrediction9 prediction,
    in float x[9],
    in float target[9],
    in float scale[9],
    out float residual[9]
)
{
    float norm = 0.0;
    for (int row = 0; row < 6; ++row)
    {
        residual[row] =
            (prediction.conserved[row] - target[row]) / scale[row];
        norm = max(norm, abs(residual[row]));
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        residual[6 + axis] =
            (x[6 + axis] - x[2 + axis] - target[6 + axis])
            / scale[6 + axis];
        norm = max(norm, abs(residual[6 + axis]));
    }
    return norm;
}

void theory33DragKernel(
    int point,
    int modelMode,
    float dt,
    float lapse,
    float relaxationTime,
    ITensor<float, 2> primitiveInput,
    IWTensor<float, 2> primitiveOutput,
    IWTensor<float, 2> diagnostics
)
{
    float initial[9] = {};
    for (int index = 0; index < 9; ++index)
        initial[index] = primitiveInput[point, index];
    VTPrediction9 before = vtPredict9(initial, modelMode, -1);
    if (before.relativeGamma - 1.0 <= 1.0e-3)
    {
        for (int index = 0; index < 9; ++index)
            primitiveOutput[point, index] = initial[index];
        diagnostics[point, 0] = before.relativeGamma;
        diagnostics[point, 1] = before.relativeGamma;
        diagnostics[point, 2] = 0.0;
        diagnostics[point, 3] = 0.0;
        diagnostics[point, 4] = 1.0;
        return;
    }

    float decay = exp(
        -dt * max(lapse, 0.0) / max(relaxationTime, VT_EPSILON)
    );
    float target[9] = {};
    float scale[9] = {};
    for (int row = 0; row < 6; ++row)
        target[row] = before.conserved[row];
    for (int axis = 0; axis < 3; ++axis)
    {
        target[6 + axis] =
            decay * (initial[6 + axis] - initial[2 + axis]);
    }
    float energyScale = max(
        max(
            abs(before.conserved[2]),
            max(abs(before.conserved[3]), abs(before.conserved[4]))
        ),
        max(abs(before.conserved[5]), 1.0e-8)
    );
    scale[0] = max(abs(before.conserved[0]), 1.0e-8);
    scale[1] = max(abs(before.conserved[1]), 1.0e-8);
    for (int row = 2; row < 6; ++row)
        scale[row] = energyScale;
    scale[6] = 1.0;
    scale[7] = 1.0;
    scale[8] = 1.0;

    float working[9] = {};
    for (int index = 0; index < 9; ++index)
        working[index] = initial[index];
    int phaseOverride = modelMode == 1
        ? (before.phaseTwo ? 1 : 0)
        : -1;
    float finalNorm = 1.0e30;
    int phaseCrossings = 0;
    [MaxIters(3)]
    for (int transition = 0; transition < 3; ++transition)
    {
        [MaxIters(24)]
        for (int iteration = 0; iteration < 24; ++iteration)
        {
            VTPrediction9 prediction = vtPredict9(
                working, modelMode, phaseOverride
            );
            float residual[9] = {};
            finalNorm = vtDragResidual(
                prediction, working, target, scale, residual
            );
            if (finalNorm < 3.0e-5)
                break;
            float jacobian[81] = {};
            for (int column = 0; column < 9; ++column)
            {
                float step =
                    3.0e-4 * max(1.0, abs(working[column]));
                float plus[9] = {};
                float minus[9] = {};
                for (int index = 0; index < 9; ++index)
                {
                    plus[index] = working[index];
                    minus[index] = working[index];
                }
                plus[column] += step;
                minus[column] -= step;
                VTPrediction9 plusPrediction = vtPredict9(
                    plus, modelMode, phaseOverride
                );
                VTPrediction9 minusPrediction = vtPredict9(
                    minus, modelMode, phaseOverride
                );
                float plusResidual[9] = {};
                float minusResidual[9] = {};
                vtDragResidual(
                    plusPrediction, plus, target, scale, plusResidual
                );
                vtDragResidual(
                    minusPrediction, minus, target, scale, minusResidual
                );
                for (int row = 0; row < 9; ++row)
                {
                    jacobian[row * 9 + column] =
                        (plusResidual[row] - minusResidual[row])
                        / (2.0 * step);
                }
            }
            float rhs[9] = {};
            float delta[9] = {};
            for (int row = 0; row < 9; ++row)
                rhs[row] = -residual[row];
            if (!vtSolve9(jacobian, rhs, delta))
                break;
            bool accepted = false;
            float factor = 1.0;
            [MaxIters(12)]
            for (int lineSearch = 0; lineSearch < 12; ++lineSearch)
            {
                float trial[9] = {};
                for (int index = 0; index < 9; ++index)
                    trial[index] = working[index] + factor * delta[index];
                trial[0] = clamp(trial[0], -20.0, 4.0);
                trial[1] = clamp(trial[1], -20.0, 20.0);
                trial[5] = clamp(trial[5], -20.0, 4.0);
                VTPrediction9 trialPrediction = vtPredict9(
                    trial, modelMode, phaseOverride
                );
                float trialResidual[9] = {};
                float trialNorm = vtDragResidual(
                    trialPrediction,
                    trial,
                    target,
                    scale,
                    trialResidual
                );
                if (trialNorm < finalNorm)
                {
                    for (int index = 0; index < 9; ++index)
                        working[index] = trial[index];
                    accepted = true;
                    break;
                }
                factor *= 0.5;
            }
            if (!accepted)
                break;
        }
        VTPrediction9 recovered = vtPredict9(
            working, modelMode, phaseOverride
        );
        int finalPhase = recovered.switchingMargin >= 0.0 ? 1 : 0;
        if (modelMode != 1 || finalPhase == phaseOverride)
            break;
        phaseOverride = finalPhase;
        phaseCrossings += 1;
    }

    VTPrediction9 after = vtPredict9(
        working, modelMode, phaseOverride
    );
    float finalResidual[9] = {};
    finalNorm = vtDragResidual(
        after, working, target, scale, finalResidual
    );
    float entropyIncrease = working[1] - initial[1];
    bool valid =
        isfinite(finalNorm)
        && finalNorm < 5.0e-4
        && entropyIncrease >= -2.0e-5
        && after.relativeGamma <= before.relativeGamma + 1.0e-5;
    for (int index = 0; index < 9; ++index)
        primitiveOutput[point, index] = working[index];
    diagnostics[point, 0] = before.relativeGamma;
    diagnostics[point, 1] = after.relativeGamma;
    diagnostics[point, 2] = entropyIncrease;
    diagnostics[point, 3] = finalNorm;
    diagnostics[point, 4] = valid ? 1.0 : 0.0;
    diagnostics[point, 5] = float(phaseCrossings);
    diagnostics[point, 6] = decay;
    diagnostics[point, 7] =
        after.temperature * exp(working[0]) * max(entropyIncrease, 0.0);
}

void vtFieldTensor(in float magnetic[3], out float field[9])
{
    for (int index = 0; index < 9; ++index)
        field[index] = 0.0;
    field[1 * 3 + 2] = magnetic[0];
    field[2 * 3 + 1] = -magnetic[0];
    field[2 * 3 + 0] = magnetic[1];
    field[0 * 3 + 2] = -magnetic[1];
    field[0 * 3 + 1] = magnetic[2];
    field[1 * 3 + 0] = -magnetic[2];
}

void vtSingleVectorStress(
    in float potential[3],
    in float momentumInput[3],
    in float magnetic[3],
    float phi,
    float kinetic,
    float massSquared,
    out float rho,
    out float momentum[3],
    out float stress[9]
)
{
    float electric[3] = {};
    for (int axis = 0; axis < 3; ++axis)
        electric[axis] = -momentumInput[axis] / kinetic;
    float field[9] = {};
    vtFieldTensor(magnetic, field);
    float electricSquared = vtDot3(electric, electric);
    float potentialSquared = vtDot3(potential, potential);
    float fieldSquared = 0.0;
    for (int index = 0; index < 9; ++index)
        fieldSquared += field[index] * field[index];
    rho = kinetic * (
        0.5 * electricSquared
        + 0.25 * fieldSquared
        + 0.5 * massSquared * (phi * phi + potentialSquared)
    );
    for (int axis = 0; axis < 3; ++axis)
    {
        momentum[axis] = 0.0;
        for (int inner = 0; inner < 3; ++inner)
            momentum[axis] += electric[inner] * field[axis * 3 + inner];
        momentum[axis] =
            kinetic
            * (
                momentum[axis]
                + massSquared * phi * potential[axis]
            );
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            float magneticTensor = 0.0;
            for (int inner = 0; inner < 3; ++inner)
            {
                magneticTensor +=
                    field[row * 3 + inner]
                    * field[column * 3 + inner];
            }
            stress[row * 3 + column] = kinetic * (
                -electric[row] * electric[column]
                + magneticTensor
                + (row == column
                    ? 0.5 * electricSquared - 0.25 * fieldSquared
                    : 0.0)
                + massSquared
                * (
                    potential[row] * potential[column]
                    - (row == column
                        ? 0.5 * (potentialSquared - phi * phi)
                        : 0.0)
                )
            );
        }
    }
}

void theory33VectorStressKernel(
    int point,
    ITensor<float, 2> primitive,
    ITensor<float, 2> vectorInput,
    IWTensor<float, 2> output
)
{
    float a[3] = {};
    float piA[3] = {};
    float b[3] = {};
    float piB[3] = {};
    float magneticA[3] = {};
    float magneticB[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        a[axis] = vectorInput[point, axis];
        piA[axis] = vectorInput[point, 3 + axis];
        b[axis] = vectorInput[point, 6 + axis];
        piB[axis] = vectorInput[point, 9 + axis];
        magneticA[axis] = vectorInput[point, 14 + axis];
        magneticB[axis] = vectorInput[point, 17 + axis];
    }
    float x[9];
    [ForceUnroll]
    for (int component = 0; component < 9; ++component)
    {
        x[component] = primitive[point, component];
    }
    VTPrediction9 material = vtPredict9(x, 0, -1);
    float targetCharge = material.conserved[1];
    float rightA = vectorInput[point, 12];
    float rightB = vectorInput[point, 13] + targetCharge;
    float phiA = (8.0 / 3.0) * rightA + (4.0 / 3.0) * rightB;
    float phiB = (4.0 / 3.0) * rightA + (8.0 / 3.0) * rightB;

    float rhoA;
    float momentumA[3] = {};
    float stressA[9] = {};
    vtSingleVectorStress(
        a, piA, magneticA, phiA, 1.0, 0.25,
        rhoA, momentumA, stressA
    );
    float rhoB;
    float momentumB[3] = {};
    float stressB[9] = {};
    vtSingleVectorStress(
        b, piB, magneticB, phiB, 1.0, 0.25,
        rhoB, momentumB, stressB
    );
    float difference[3] = {};
    for (int axis = 0; axis < 3; ++axis)
        difference[axis] = a[axis] - b[axis];
    float phiDifference = phiA - phiB;
    float differenceSquared = vtDot3(difference, difference);
    float mixingRho =
        0.125 * (phiDifference * phiDifference + differenceSquared);
    output[point, 0] = rhoA + rhoB + mixingRho;
    for (int axis = 0; axis < 3; ++axis)
    {
        float mixingMomentum =
            0.25 * phiDifference * difference[axis];
        output[point, 1 + axis] =
            momentumA[axis] + momentumB[axis] + mixingMomentum;
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            float mixingStress = 0.25 * (
                difference[row] * difference[column]
                - (row == column
                    ? 0.5
                    * (differenceSquared - phiDifference * phiDifference)
                    : 0.0)
            );
            output[point, 4 + row * 3 + column] =
                stressA[row * 3 + column]
                + stressB[row * 3 + column]
                + mixingStress;
        }
    }
    output[point, 13] = phiA;
    output[point, 14] = phiB;
}

// Superseded single-dispatch prototype retained out of compilation for
// provenance. The qualified path below uses isolated A/B passes so no
// driver can alias their fixed-array temporaries.
#if 0
void vtSingleVectorStressCurved(
    in float potential[3],
    in float momentumInput[3],
    in float magnetic[3],
    float phi,
    in float metricInput[9],
    in float inverseMetricInput[9],
    float sqrtDeterminant,
    out float rho,
    out float momentum[3],
    out float stress[9]
)
{
    float electricUp[3] = {};
    float electricDown[3] = {};
    float potentialUp[3] = {};
    for (int row = 0; row < 3; ++row)
    {
        electricUp[row] =
            -momentumInput[row] / sqrtDeterminant;
        for (int column = 0; column < 3; ++column)
        {
            electricDown[row] +=
                metricInput[row * 3 + column]
                * electricUp[column];
            potentialUp[row] +=
                inverseMetricInput[row * 3 + column]
                * potential[column];
        }
    }
    float field[9] = {};
    vtFieldTensor(magnetic, field);
    float metric[9] = {};
    float inverseMetric[9] = {};
    for (int index = 0; index < 9; ++index)
    {
        metric[index] = metricInput[index];
        inverseMetric[index] = inverseMetricInput[index];
    }
    float electricSquared = 0.0;
    float potentialSquared = 0.0;
    float fieldSquared = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        electricSquared += electricDown[row] * electricUp[row];
        potentialSquared += potential[row] * potentialUp[row];
        for (int column = 0; column < 3; ++column)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                for (int second = 0; second < 3; ++second)
                {
                    fieldSquared +=
                        field[row * 3 + column]
                        * inverseMetric[row * 3 + inner]
                        * inverseMetric[column * 3 + second]
                        * field[inner * 3 + second];
                }
            }
        }
    }
    rho =
        0.5 * electricSquared
        + 0.25 * fieldSquared
        + 0.125 * (phi * phi + potentialSquared);
    for (int row = 0; row < 3; ++row)
    {
        momentum[row] = 0.25 * phi * potential[row];
        for (int column = 0; column < 3; ++column)
            momentum[row] += electricUp[column] * field[row * 3 + column];
        for (int column = 0; column < 3; ++column)
        {
            float magneticTensor = 0.0;
            for (int first = 0; first < 3; ++first)
            {
                for (int second = 0; second < 3; ++second)
                {
                    magneticTensor +=
                        field[row * 3 + first]
                        * inverseMetric[first * 3 + second]
                        * field[column * 3 + second];
                }
            }
            stress[row * 3 + column] =
                -electricDown[row] * electricDown[column]
                + magneticTensor
                + metric[row * 3 + column]
                * (0.5 * electricSquared - 0.25 * fieldSquared)
                + 0.25
                * (
                    potential[row] * potential[column]
                    - 0.5
                    * metric[row * 3 + column]
                    * (potentialSquared - phi * phi)
                );
        }
    }
}

void vtSingleVectorStressCurvedStable(
    float3 potential,
    float3 momentumInput,
    float3 magnetic,
    float phi,
    float3x3 metric,
    float3x3 inverseMetric,
    float sqrtDeterminant,
    out float rho,
    out float3 momentum,
    out float3x3 stress
)
{
    float3 electricUp;
    float3 electricDown;
    float3 potentialUp;
    for (int row = 0; row < 3; ++row)
    {
        electricUp[row] = -momentumInput[row] / sqrtDeterminant;
        electricDown[row] = 0.0;
        potentialUp[row] = 0.0;
        for (int column = 0; column < 3; ++column)
        {
            electricDown[row] +=
                metric[row][column] * electricUp[column];
            potentialUp[row] +=
                inverseMetric[row][column] * potential[column];
        }
    }
    float3x3 field;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
            field[row][column] = 0.0;
    }
    field[1][2] = magnetic[0];
    field[2][1] = -magnetic[0];
    field[2][0] = magnetic[1];
    field[0][2] = -magnetic[1];
    field[0][1] = magnetic[2];
    field[1][0] = -magnetic[2];
    float electricSquared = 0.0;
    float potentialSquared = 0.0;
    float fieldSquared = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        electricSquared += electricDown[row] * electricUp[row];
        potentialSquared += potential[row] * potentialUp[row];
        for (int column = 0; column < 3; ++column)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                for (int second = 0; second < 3; ++second)
                {
                    fieldSquared +=
                        field[row][column]
                        * inverseMetric[row][inner]
                        * inverseMetric[column][second]
                        * field[inner][second];
                }
            }
        }
    }
    rho =
        0.5 * electricSquared
        + 0.25 * fieldSquared
        + 0.125 * (phi * phi + potentialSquared);
    for (int row = 0; row < 3; ++row)
    {
        momentum[row] = 0.25 * phi * potential[row];
        for (int column = 0; column < 3; ++column)
            momentum[row] += electricUp[column] * field[row][column];
        for (int column = 0; column < 3; ++column)
        {
            float magneticTensor = 0.0;
            for (int first = 0; first < 3; ++first)
            {
                for (int second = 0; second < 3; ++second)
                {
                    magneticTensor +=
                        field[row][first]
                        * inverseMetric[first][second]
                        * field[column][second];
                }
            }
            stress[row][column] =
                -electricDown[row] * electricDown[column]
                + magneticTensor
                + metric[row][column]
                * (0.5 * electricSquared - 0.25 * fieldSquared)
                + 0.25
                * (
                    potential[row] * potential[column]
                    - 0.5
                    * metric[row][column]
                    * (potentialSquared - phi * phi)
                );
        }
    }
}

void theory33CurvedVectorStressKernel(
    int point,
    ITensor<float, 2> primitive,
    ITensor<float, 2> vectorInput,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> output
)
{
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float metricForStress[9] = {};
    float inverseForStress[9] = {};
    for (int index = 0; index < 9; ++index)
    {
        metricForStress[index] = geometryValues[index];
        inverseForStress[index] = geometry.inverseMetric[index];
    }
    float a[3] = {};
    float piA[3] = {};
    float b[3] = {};
    float piB[3] = {};
    float magneticA[3] = {};
    float magneticB[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        a[axis] = vectorInput[point, axis];
        piA[axis] = vectorInput[point, 3 + axis];
        b[axis] = vectorInput[point, 6 + axis];
        piB[axis] = vectorInput[point, 9 + axis];
        magneticA[axis] = vectorInput[point, 14 + axis];
        magneticB[axis] = vectorInput[point, 17 + axis];
    }
    float x[9] = {};
    for (int component = 0; component < 9; ++component)
        x[component] = primitive[point, component];
    VTPrincipal9 material = vtPrincipal9(x, geometry, 0, 0, -1);
    float targetCharge =
        material.carrierEulerian / geometry.sqrtDeterminant;
    float rightA =
        vectorInput[point, 12] / geometry.sqrtDeterminant;
    float rightB =
        vectorInput[point, 13] / geometry.sqrtDeterminant
        + targetCharge;
    float phiA = (8.0 / 3.0) * rightA + (4.0 / 3.0) * rightB;
    float phiB = (4.0 / 3.0) * rightA + (8.0 / 3.0) * rightB;
    float3x3 metricMatrix;
    float3x3 inverseMatrix;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            metricMatrix[row][column] =
                metricForStress[row * 3 + column];
            inverseMatrix[row][column] =
                inverseForStress[row * 3 + column];
        }
    }
    float3 aVector;
    float3 piAVector;
    float3 magneticAVector;
    float3 bVector;
    float3 piBVector;
    float3 magneticBVector;
    for (int axis = 0; axis < 3; ++axis)
    {
        aVector[axis] = a[axis];
        piAVector[axis] = piA[axis];
        magneticAVector[axis] = magneticA[axis];
        bVector[axis] = b[axis];
        piBVector[axis] = piB[axis];
        magneticBVector[axis] = magneticB[axis];
    }
    float rhoA;
    float3 momentumA;
    float3x3 stressA;
    vtSingleVectorStressCurvedStable(
        aVector,
        piAVector,
        magneticAVector,
        phiA,
        metricMatrix,
        inverseMatrix,
        geometry.sqrtDeterminant,
        rhoA,
        momentumA,
        stressA
    );
    float storedRhoA = rhoA;
    float3 storedMomentumA = momentumA;
    float3x3 storedStressA = stressA;
    float rhoB;
    float3 momentumB;
    float3x3 stressB;
    vtSingleVectorStressCurvedStable(
        bVector,
        piBVector,
        magneticBVector,
        phiB,
        metricMatrix,
        inverseMatrix,
        geometry.sqrtDeterminant,
        rhoB,
        momentumB,
        stressB
    );
    // Rebuild field inputs after the two struct/array-heavy stress calls.
    for (int axis = 0; axis < 3; ++axis)
    {
        a[axis] = vectorInput[point, axis];
        b[axis] = vectorInput[point, 6 + axis];
    }
    float difference[3] = {};
    float differenceUp[3] = {};
    for (int row = 0; row < 3; ++row)
    {
        difference[row] = a[row] - b[row];
        for (int column = 0; column < 3; ++column)
        {
            differenceUp[row] +=
                geometry.inverseMetric[row * 3 + column]
                * difference[column];
        }
    }
    float differenceSquared = vtDot3(difference, differenceUp);
    float phiDifference = phiA - phiB;
    output[point, 0] =
        storedRhoA
        +
        rhoB
        + 0.125 * (phiDifference * phiDifference + differenceSquared);
    for (int axis = 0; axis < 3; ++axis)
    {
        output[point, 1 + axis] =
            storedMomentumA[axis]
            +
            momentumB[axis]
            + 0.25 * phiDifference * difference[axis];
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            float mixingStress = 0.25 * (
                difference[row] * difference[column]
                - 0.5
                * geometry.metric[row * 3 + column]
                * (
                    differenceSquared
                    - phiDifference * phiDifference
                )
            );
            output[point, 4 + row * 3 + column] =
                storedStressA[row][column]
                + stressB[row][column]
                + mixingStress;
        }
    }
    output[point, 13] = phiA;
    output[point, 14] = phiB;
}

void theory33CurvedVectorAuditKernel(
    int point,
    ITensor<float, 2> vectorInput,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> audit
)
{
    float values[13] = {};
    for (int index = 0; index < 13; ++index)
        values[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(values);
    float field[9] = {};
    float magnetic[3] = {};
    for (int axis = 0; axis < 3; ++axis)
        magnetic[axis] = vectorInput[point, 14 + axis];
    vtFieldTensor(magnetic, field);
    float fieldSquared = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                for (int second = 0; second < 3; ++second)
                {
                    fieldSquared +=
                        field[row * 3 + column]
                        * geometry.inverseMetric[row * 3 + inner]
                        * geometry.inverseMetric[column * 3 + second]
                        * field[inner * 3 + second];
                }
            }
        }
    }
    audit[point, 0] = fieldSquared;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            float magneticTensor = 0.0;
            for (int first = 0; first < 3; ++first)
            {
                for (int second = 0; second < 3; ++second)
                {
                    magneticTensor +=
                        field[row * 3 + first]
                        * geometry.inverseMetric[first * 3 + second]
                        * field[column * 3 + second];
                }
            }
            audit[point, 1 + row * 3 + column] = magneticTensor;
        }
    }
}
#endif

void theory33SingleCurvedVectorStressKernel(
    int point,
    int fieldIndex,
    ITensor<float, 2> vectorInput,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> output
)
{
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    int potentialOffset = fieldIndex == 0 ? 0 : 6;
    int momentumOffset = fieldIndex == 0 ? 3 : 9;
    int magneticOffset = fieldIndex == 0 ? 14 : 17;
    float potential[3] = {};
    float electricUp[3] = {};
    float electricDown[3] = {};
    float potentialUp[3] = {};
    float magnetic[3] = {};
    for (int row = 0; row < 3; ++row)
    {
        potential[row] =
            vectorInput[point, potentialOffset + row];
        electricUp[row] =
            -vectorInput[point, momentumOffset + row]
            / geometry.sqrtDeterminant;
        magnetic[row] = vectorInput[point, magneticOffset + row];
        for (int column = 0; column < 3; ++column)
        {
            electricDown[row] +=
                geometry.metric[row * 3 + column]
                * electricUp[column];
            potentialUp[row] +=
                geometry.inverseMetric[row * 3 + column]
                * potential[column];
        }
    }
    float field[9] = {};
    vtFieldTensor(magnetic, field);
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 stableGeometry = vtGeometry3(geometryValues);
    float stableMetric[9] = {};
    float stableInverse[9] = {};
    for (int index = 0; index < 9; ++index)
    {
        stableMetric[index] = geometryValues[index];
        stableInverse[index] = stableGeometry.inverseMetric[index];
    }
    for (int row = 0; row < 3; ++row)
    {
        electricDown[row] = 0.0;
        potentialUp[row] = 0.0;
        for (int column = 0; column < 3; ++column)
        {
            electricDown[row] +=
                stableMetric[row * 3 + column] * electricUp[column];
            potentialUp[row] +=
                stableInverse[row * 3 + column] * potential[column];
        }
    }
    float electricSquared = vtDot3(electricDown, electricUp);
    float potentialSquared = vtDot3(potential, potentialUp);
    float fieldSquared = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                for (int second = 0; second < 3; ++second)
                {
                    fieldSquared +=
                        field[row * 3 + column]
                        * stableInverse[row * 3 + inner]
                        * stableInverse[column * 3 + second]
                        * field[inner * 3 + second];
                }
            }
        }
    }
    // The scalar potential is supplied by the combine kernel because it
    // depends on both Gauss sectors. Store the phi-independent pieces plus
    // the coefficients needed to add the mass term there.
    output[point, 0] =
        0.5 * electricSquared
        + 0.25 * fieldSquared
        + 0.125 * potentialSquared;
    for (int row = 0; row < 3; ++row)
    {
        float momentum = 0.0;
        for (int column = 0; column < 3; ++column)
            momentum += electricUp[column] * field[row * 3 + column];
        output[point, 1 + row] = momentum;
        for (int column = 0; column < 3; ++column)
        {
            float magneticTensor = 0.0;
            for (int first = 0; first < 3; ++first)
            {
                for (int second = 0; second < 3; ++second)
                {
                    magneticTensor +=
                        field[row * 3 + first]
                        * stableInverse[first * 3 + second]
                        * field[column * 3 + second];
                }
            }
            output[point, 4 + row * 3 + column] =
                -electricDown[row] * electricDown[column]
                + magneticTensor
                + stableMetric[row * 3 + column]
                * (0.5 * electricSquared - 0.25 * fieldSquared)
                + 0.25
                * (
                    potential[row] * potential[column]
                    - 0.5
                    * stableMetric[row * 3 + column]
                    * potentialSquared
                );
        }
    }
}

void theory33CombineCurvedVectorStressKernel(
    int point,
    ITensor<float, 2> primitive,
    ITensor<float, 2> vectorInput,
    ITensor<float, 2> geometryInput,
    ITensor<float, 2> stressA,
    ITensor<float, 2> stressB,
    IWTensor<float, 2> output
)
{
    float geometryValues[13] = {};
    float x[9] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    for (int index = 0; index < 9; ++index)
        x[index] = primitive[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    VTPrincipal9 material = vtPrincipal9(x, geometry, 0, 0, -1);
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 stableGeometry = vtGeometry3(geometryValues);
    float rightA =
        vectorInput[point, 12] / stableGeometry.sqrtDeterminant;
    float rightB =
        vectorInput[point, 13] / stableGeometry.sqrtDeterminant
        + material.carrierEulerian / stableGeometry.sqrtDeterminant;
    float phiA = (8.0 / 3.0) * rightA + (4.0 / 3.0) * rightB;
    float phiB = (4.0 / 3.0) * rightA + (8.0 / 3.0) * rightB;
    float difference[3] = {};
    float differenceUp[3] = {};
    for (int row = 0; row < 3; ++row)
    {
        difference[row] =
            vectorInput[point, row] - vectorInput[point, 6 + row];
    }
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 mixingGeometry = vtGeometry3(geometryValues);
    float mixingMetric[9] = {};
    float mixingInverse[9] = {};
    for (int index = 0; index < 9; ++index)
    {
        mixingMetric[index] = geometryValues[index];
        mixingInverse[index] = mixingGeometry.inverseMetric[index];
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            differenceUp[row] +=
                mixingInverse[row * 3 + column] * difference[column];
        }
    }
    float differenceSquared = vtDot3(difference, differenceUp);
    float phiDifference = phiA - phiB;
    output[point, 0] =
        stressA[point, 0]
        + stressB[point, 0]
        + 0.125 * (
            phiA * phiA
            + phiB * phiB
            + phiDifference * phiDifference
            + differenceSquared
        );
    for (int row = 0; row < 3; ++row)
    {
        output[point, 1 + row] =
            stressA[point, 1 + row]
            + stressB[point, 1 + row]
            + 0.25
            * (
                phiA * vectorInput[point, row]
                + phiB * vectorInput[point, 6 + row]
                + phiDifference * difference[row]
            );
        for (int column = 0; column < 3; ++column)
        {
            output[point, 4 + row * 3 + column] =
                stressA[point, 4 + row * 3 + column]
                + stressB[point, 4 + row * 3 + column]
                + 0.125
                * mixingMetric[row * 3 + column]
                * (phiA * phiA + phiB * phiB)
                + 0.25
                * (
                    difference[row] * difference[column]
                    - 0.5
                    * mixingMetric[row * 3 + column]
                    * (
                        differenceSquared
                        - phiDifference * phiDifference
                    )
                );
        }
    }
    output[point, 13] = phiA;
    output[point, 14] = phiB;
}

void theory33ComposeNineFieldKernel(
    int point,
    int modelMode,
    ITensor<float, 2> primitive,
    ITensor<float, 1> tracer,
    ITensor<float, 2> vectorStress,
    IWTensor<float, 2> conserved
)
{
    float x[9] = {};
    for (int index = 0; index < 9; ++index)
        x[index] = primitive[point, index];
    VTPrediction9 material = vtPredict9(x, modelMode, -1);
    conserved[point, 0] = material.conserved[0];
    conserved[point, 1] = material.conserved[1];
    for (int axis = 0; axis < 3; ++axis)
    {
        conserved[point, 2 + axis] =
            material.conserved[2 + axis]
            + vectorStress[point, 1 + axis];
    }
    conserved[point, 5] =
        material.conserved[5] + vectorStress[point, 0];
    for (int axis = 0; axis < 3; ++axis)
        conserved[point, 6 + axis] = material.conserved[6 + axis];
    conserved[point, 9] =
        material.conserved[0] * material.entropy;
    conserved[point, 10] =
        material.conserved[0] * tracer[point];
}

struct VTGeometry3
{
    float metric[9];
    float inverseMetric[9];
    float sqrtDeterminant;
    float lapse;
    float shift[3];
    bool valid;
};

VTGeometry3 vtGeometry3(in float input[13])
{
    VTGeometry3 result = {};
    for (int index = 0; index < 9; ++index)
        result.metric[index] = input[index];
    result.lapse = input[9];
    result.shift[0] = input[10];
    result.shift[1] = input[11];
    result.shift[2] = input[12];
    float a = result.metric[0];
    float b = result.metric[1];
    float c = result.metric[2];
    float d = result.metric[3];
    float e = result.metric[4];
    float f = result.metric[5];
    float g = result.metric[6];
    float h = result.metric[7];
    float i = result.metric[8];
    float determinant =
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g);
    result.valid =
        isfinite(determinant)
        && determinant > 1.0e-12
        && result.lapse > 0.0;
    float inverseDeterminant = 1.0 / max(determinant, 1.0e-12);
    result.inverseMetric[0] = (e * i - f * h) * inverseDeterminant;
    result.inverseMetric[1] = (c * h - b * i) * inverseDeterminant;
    result.inverseMetric[2] = (b * f - c * e) * inverseDeterminant;
    result.inverseMetric[3] = (f * g - d * i) * inverseDeterminant;
    result.inverseMetric[4] = (a * i - c * g) * inverseDeterminant;
    result.inverseMetric[5] = (c * d - a * f) * inverseDeterminant;
    result.inverseMetric[6] = (d * h - e * g) * inverseDeterminant;
    result.inverseMetric[7] = (b * g - a * h) * inverseDeterminant;
    result.inverseMetric[8] = (a * e - b * d) * inverseDeterminant;
    result.sqrtDeterminant = sqrt(max(determinant, 1.0e-12));
    return result;
}

float vtMetricDot3(
    in float left[3],
    in float metric[9],
    in float right[3]
)
{
    float value = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            value +=
                left[row]
                * metric[row * 3 + column]
                * right[column];
        }
    }
    return value;
}

struct VTPrincipal9
{
    float conserved[9];
    float flux[9];
    float chiZero;
    float chiDown[3];
    float carrierEulerian;
    float carrierTransport[3];
    bool phaseTwo;
    float switchingMargin;
    bool valid;
};

VTPrincipal9 vtPrincipal9(
    in float x[9],
    in VTGeometry3 geometry,
    int axis,
    int modelMode,
    int phaseOverride
)
{
    VTPrincipal9 result = {};
    float n = exp(clamp(x[0], -20.0, 20.0));
    float entropy = clamp(x[1], -20.0, 20.0);
    float qN[3] = {};
    float qD[3] = {};
    for (int component = 0; component < 3; ++component)
    {
        qN[component] = x[2 + component];
        qD[component] = x[6 + component];
    }
    float d = exp(clamp(x[5], -20.0, 20.0));
    float wN = sqrt(
        max(1.0 + vtMetricDot3(qN, geometry.metric, qN), 1.0)
    );
    float wD = sqrt(
        max(1.0 + vtMetricDot3(qD, geometry.metric, qD), 1.0)
    );
    float velocityN[3] = {};
    float velocityD[3] = {};
    float velocityNDown[3] = {};
    float velocityDDown[3] = {};
    for (int component = 0; component < 3; ++component)
    {
        velocityN[component] = qN[component] / wN;
        velocityD[component] = qD[component] / wD;
        for (int inner = 0; inner < 3; ++inner)
        {
            velocityNDown[component] +=
                geometry.metric[component * 3 + inner]
                * velocityN[inner];
            velocityDDown[component] +=
                geometry.metric[component * 3 + inner]
                * velocityD[inner];
        }
    }
    float relativeGamma = max(
        wN * wD - vtMetricDot3(qN, geometry.metric, qD),
        1.0
    );
    float cross = n * d * relativeGamma;
    VTConstitutive material = vtConstitutive(
        n, d, cross, entropy, modelMode, phaseOverride
    );
    // Rebuild the small vector temporaries after the struct-returning
    // constitutive call. This keeps their storage independent on Vulkan
    // drivers that aggressively reuse function-local array slots.
    for (int component = 0; component < 3; ++component)
    {
        qN[component] = x[2 + component];
        qD[component] = x[6 + component];
        velocityN[component] = qN[component] / wN;
        velocityD[component] = qD[component] / wD;
        velocityNDown[component] = 0.0;
        velocityDDown[component] = 0.0;
        for (int inner = 0; inner < 3; ++inner)
        {
            velocityNDown[component] +=
                geometry.metric[component * 3 + inner]
                * velocityN[inner];
            velocityDDown[component] +=
                geometry.metric[component * 3 + inner]
                * velocityD[inner];
        }
    }
    float nW = n * wN;
    float dW = d * wD;
    float densitizedN = geometry.sqrtDeterminant * nW;
    float densitizedD = geometry.sqrtDeterminant * dW;
    float momentum[3] = {};
    float chiDown[3] = {};
    for (int component = 0; component < 3; ++component)
    {
        momentum[component] =
            material.bN * nW * nW * velocityNDown[component]
            + material.bD * dW * dW * velocityDDown[component]
            + material.entrainment
            * nW
            * dW
            * (
                velocityNDown[component]
                + velocityDDown[component]
            );
        chiDown[component] =
            material.bD * dW * velocityDDown[component]
            + material.entrainment * nW * velocityNDown[component];
    }
    float energy =
        -material.generalizedPressure
        + material.bN * nW * nW
        + material.bD * dW * dW
        + 2.0 * material.entrainment * nW * dW;
    result.conserved[0] = densitizedN;
    result.conserved[1] = densitizedD;
    for (int component = 0; component < 3; ++component)
    {
        result.conserved[2 + component] =
            geometry.sqrtDeterminant * momentum[component];
        result.conserved[6 + component] =
            densitizedD * chiDown[component];
    }
    result.conserved[5] = geometry.sqrtDeterminant * energy;

    float transportN =
        geometry.lapse * velocityN[axis] - geometry.shift[axis];
    float transportD =
        geometry.lapse * velocityD[axis] - geometry.shift[axis];
    result.flux[0] = densitizedN * transportN;
    result.flux[1] = densitizedD * transportD;
    for (int component = 0; component < 3; ++component)
    {
        float stressMixed = 0.0;
        for (int inner = 0; inner < 3; ++inner)
        {
            float stressDown =
                material.generalizedPressure
                * geometry.metric[inner * 3 + component]
                + material.bN
                * nW
                * nW
                * velocityNDown[inner]
                * velocityNDown[component]
                + material.bD
                * dW
                * dW
                * velocityDDown[inner]
                * velocityDDown[component]
                + material.entrainment
                * nW
                * dW
                * (
                    velocityNDown[inner]
                    * velocityDDown[component]
                    + velocityDDown[inner]
                    * velocityNDown[component]
                );
            stressMixed +=
                geometry.inverseMetric[axis * 3 + inner] * stressDown;
        }
        result.flux[2 + component] =
            geometry.sqrtDeterminant
            * (
                geometry.lapse * stressMixed
                - geometry.shift[axis] * momentum[component]
            );
        result.flux[6 + component] =
            result.conserved[6 + component] * transportD;
    }
    float momentumUpAxis = 0.0;
    for (int component = 0; component < 3; ++component)
    {
        momentumUpAxis +=
            geometry.inverseMetric[axis * 3 + component]
            * momentum[component];
    }
    result.flux[5] =
        geometry.sqrtDeterminant
        * (
            geometry.lapse * momentumUpAxis
            - geometry.shift[axis] * energy
        );
    float chiNormal =
        material.bD * dW + material.entrainment * nW;
    result.chiZero = -geometry.lapse * chiNormal;
    for (int component = 0; component < 3; ++component)
    {
        result.chiZero += geometry.shift[component] * chiDown[component];
        result.chiDown[component] = chiDown[component];
        result.carrierTransport[component] =
            geometry.lapse * velocityD[component] - geometry.shift[component];
    }
    result.carrierEulerian = densitizedD;
    result.phaseTwo = material.phaseTwo;
    result.switchingMargin = material.switchingMargin;
    result.valid =
        geometry.valid
        && isfinite(energy)
        && densitizedN > VT_EPSILON
        && densitizedD > VT_EPSILON;
    return result;
}

void theory33FullPrincipalKernel(
    int point,
    int axis,
    int modelMode,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> conserved,
    IWTensor<float, 2> flux,
    IWTensor<float, 2> path,
    IWTensor<float, 1> valid
)
{
    float x[9] = {};
    float geometryValues[13] = {};
    for (int index = 0; index < 9; ++index)
        x[index] = primitive[point, index];
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    VTPrincipal9 principal = vtPrincipal9(
        x, geometry, clamp(axis, 0, 2), modelMode, -1
    );
    for (int index = 0; index < 9; ++index)
    {
        conserved[point, index] = principal.conserved[index];
        flux[point, index] = principal.flux[index];
    }
    path[point, 0] = principal.chiZero;
    path[point, 1] = principal.chiDown[clamp(axis, 0, 2)];
    path[point, 2] = principal.carrierEulerian;
    path[point, 3] = principal.carrierTransport[clamp(axis, 0, 2)];
    valid[point] = principal.valid ? 1.0 : 0.0;
}

bool vtInvert9(in float input[81], out float inverse[81])
{
    for (int column = 0; column < 9; ++column)
    {
        float rhs[9] = {};
        float solution[9] = {};
        rhs[column] = 1.0;
        if (!vtSolve9(input, rhs, solution))
            return false;
        for (int row = 0; row < 9; ++row)
            inverse[row * 9 + column] = solution[row];
    }
    return true;
}

void vtMatMul9(
    in float left[81],
    in float right[81],
    out float output[81]
)
{
    for (int row = 0; row < 9; ++row)
    {
        for (int column = 0; column < 9; ++column)
        {
            float value = 0.0;
            for (int inner = 0; inner < 9; ++inner)
            {
                value +=
                    left[row * 9 + inner]
                    * right[inner * 9 + column];
            }
            output[row * 9 + column] = value;
        }
    }
}

void vtQrEigenvalues9(in float input[81], out float values[9])
{
    float matrix[81] = {};
    for (int index = 0; index < 81; ++index)
        matrix[index] = input[index];
    [MaxIters(8)]
    for (int deflation = 0; deflation < 8; ++deflation)
    {
        int active = 9 - deflation;
        [MaxIters(72)]
        for (int iteration = 0; iteration < 72; ++iteration)
        {
            int last = active - 1;
            float diagonal = matrix[last * 9 + last];
            float previous = matrix[(last - 1) * 9 + last - 1];
            float upper = matrix[(last - 1) * 9 + last];
            float lower = matrix[last * 9 + last - 1];
            float halfDifference = 0.5 * (previous - diagonal);
            float discriminant = max(
                halfDifference * halfDifference + upper * lower,
                0.0
            );
            float denominator =
                abs(halfDifference) + sqrt(discriminant);
            float shift = diagonal;
            if (denominator > 1.0e-12)
            {
                shift -=
                    (halfDifference >= 0.0 ? 1.0 : -1.0)
                    * upper
                    * lower
                    / denominator;
            }
            float q[81] = {};
            float r[81] = {};
            for (int column = 0; column < active; ++column)
            {
                float work[9] = {};
                for (int row = 0; row < active; ++row)
                {
                    work[row] =
                        matrix[row * 9 + column]
                        - (row == column ? shift : 0.0);
                }
                for (int prior = 0; prior < column; ++prior)
                {
                    float projection = 0.0;
                    for (int row = 0; row < active; ++row)
                        projection += q[row * 9 + prior] * work[row];
                    r[prior * 9 + column] = projection;
                    for (int row = 0; row < active; ++row)
                        work[row] -= projection * q[row * 9 + prior];
                }
                float normSquared = 0.0;
                for (int row = 0; row < active; ++row)
                    normSquared += work[row] * work[row];
                float norm = sqrt(max(normSquared, 0.0));
                if (norm < 1.0e-12)
                {
                    for (int row = 0; row < active; ++row)
                        work[row] = row == column ? 1.0 : 0.0;
                    for (int prior = 0; prior < column; ++prior)
                    {
                        float projection = 0.0;
                        for (int row = 0; row < active; ++row)
                        {
                            projection +=
                                q[row * 9 + prior] * work[row];
                        }
                        for (int row = 0; row < active; ++row)
                            work[row] -= projection * q[row * 9 + prior];
                    }
                    normSquared = 0.0;
                    for (int row = 0; row < active; ++row)
                        normSquared += work[row] * work[row];
                    norm = sqrt(max(normSquared, 1.0e-24));
                }
                r[column * 9 + column] = norm;
                for (int row = 0; row < active; ++row)
                    q[row * 9 + column] = work[row] / norm;
            }
            float next[81] = {};
            for (int row = 0; row < active; ++row)
            {
                for (int column = 0; column < active; ++column)
                {
                    float value = 0.0;
                    for (int inner = 0; inner < active; ++inner)
                    {
                        value +=
                            r[row * 9 + inner]
                            * q[inner * 9 + column];
                    }
                    next[row * 9 + column] =
                        value + (row == column ? shift : 0.0);
                }
            }
            for (int row = 0; row < active; ++row)
            {
                for (int column = 0; column < active; ++column)
                    matrix[row * 9 + column] = next[row * 9 + column];
            }
            float subdiagonal = abs(matrix[last * 9 + last - 1]);
            float localScale =
                1.0
                + abs(matrix[last * 9 + last])
                + abs(matrix[(last - 1) * 9 + last - 1]);
            if (subdiagonal < 2.0e-6 * localScale)
                break;
        }
        int last = active - 1;
        matrix[last * 9 + last - 1] = 0.0;
    }
    for (int mode = 0; mode < 9; ++mode)
        values[mode] = matrix[mode * 9 + mode];
    for (int first = 0; first < 9; ++first)
    {
        for (int second = first + 1; second < 9; ++second)
        {
            if (values[second] < values[first])
            {
                float temporary = values[first];
                values[first] = values[second];
                values[second] = temporary;
            }
        }
    }
}

struct VTEigensystem9
{
    float symbol[81];
    float right[81];
    float left[81];
    float values[9];
    float maximumResidual;
    float inverseResidual;
    float conditionEstimate;
    float maximumSpeed;
    bool valid;
};

VTEigensystem9 vtEigensystem9(in float symbol[81])
{
    VTEigensystem9 result = {};
    for (int index = 0; index < 81; ++index)
        result.symbol[index] = symbol[index];
    vtQrEigenvalues9(symbol, result.values);
    for (int mode = 0; mode < 9; ++mode)
    {
        float vector[9] = {};
        vector[mode] = 1.0;
        float spectralShift =
            1.0e-4
            * (1.0 + abs(result.values[mode]))
            * (1.0 + 0.05 * float(mode));
        [MaxIters(24)]
        for (int iteration = 0; iteration < 24; ++iteration)
        {
            float shifted[81] = {};
            for (int index = 0; index < 81; ++index)
                shifted[index] = symbol[index];
            for (int diagonal = 0; diagonal < 9; ++diagonal)
            {
                shifted[diagonal * 9 + diagonal] -=
                    result.values[mode] + spectralShift;
            }
            float next[9] = {};
            if (!vtSolve9(shifted, vector, next))
                break;
            for (int previous = 0; previous < mode; ++previous)
            {
                if (
                    abs(
                        result.values[mode]
                        - result.values[previous]
                    )
                    < 2.0e-3
                )
                {
                    float projection = 0.0;
                    for (int row = 0; row < 9; ++row)
                    {
                        projection +=
                            result.right[row * 9 + previous]
                            * next[row];
                    }
                    for (int row = 0; row < 9; ++row)
                    {
                        next[row] -=
                            projection
                            * result.right[row * 9 + previous];
                    }
                }
            }
            float normSquared = 0.0;
            for (int row = 0; row < 9; ++row)
                normSquared += next[row] * next[row];
            if (normSquared < 1.0e-20)
            {
                for (int row = 0; row < 9; ++row)
                    next[row] = row == mode ? 1.0 : 0.0;
                for (int previous = 0; previous < mode; ++previous)
                {
                    if (
                        abs(
                            result.values[mode]
                            - result.values[previous]
                        )
                        < 2.0e-3
                    )
                    {
                        float projection = 0.0;
                        for (int row = 0; row < 9; ++row)
                        {
                            projection +=
                                result.right[row * 9 + previous]
                                * next[row];
                        }
                        for (int row = 0; row < 9; ++row)
                        {
                            next[row] -=
                                projection
                                * result.right[row * 9 + previous];
                        }
                    }
                }
                normSquared = 0.0;
                for (int row = 0; row < 9; ++row)
                    normSquared += next[row] * next[row];
            }
            float norm = sqrt(max(normSquared, 1.0e-30));
            for (int row = 0; row < 9; ++row)
                vector[row] = next[row] / norm;
        }
        int pivot = 0;
        float pivotMagnitude = abs(vector[0]);
        for (int row = 1; row < 9; ++row)
        {
            if (abs(vector[row]) > pivotMagnitude)
            {
                pivot = row;
                pivotMagnitude = abs(vector[row]);
            }
        }
        float signValue = vector[pivot] < 0.0 ? -1.0 : 1.0;
        for (int row = 0; row < 9; ++row)
            result.right[row * 9 + mode] = signValue * vector[row];
    }
    bool inverseValid = vtInvert9(result.right, result.left);
    result.maximumResidual = 0.0;
    result.inverseResidual = 0.0;
    float rightNorm = 0.0;
    float leftNorm = 0.0;
    result.maximumSpeed = 0.0;
    for (int mode = 0; mode < 9; ++mode)
    {
        result.maximumSpeed = max(
            result.maximumSpeed, abs(result.values[mode])
        );
        for (int row = 0; row < 9; ++row)
        {
            float applied = 0.0;
            for (int column = 0; column < 9; ++column)
            {
                applied +=
                    symbol[row * 9 + column]
                    * result.right[column * 9 + mode];
            }
            result.maximumResidual = max(
                result.maximumResidual,
                abs(
                    applied
                    - result.values[mode]
                    * result.right[row * 9 + mode]
                )
            );
        }
    }
    for (int row = 0; row < 9; ++row)
    {
        for (int column = 0; column < 9; ++column)
        {
            float product = 0.0;
            for (int inner = 0; inner < 9; ++inner)
            {
                product +=
                    result.left[row * 9 + inner]
                    * result.right[inner * 9 + column];
            }
            result.inverseResidual = max(
                result.inverseResidual,
                abs(product - (row == column ? 1.0 : 0.0))
            );
            rightNorm +=
                result.right[row * 9 + column]
                * result.right[row * 9 + column];
            leftNorm +=
                result.left[row * 9 + column]
                * result.left[row * 9 + column];
        }
    }
    result.conditionEstimate = sqrt(rightNorm * leftNorm);
    result.valid =
        inverseValid
        && isfinite(result.maximumResidual)
        && isfinite(result.maximumSpeed)
        && result.maximumResidual < 2.0e-2
        && result.inverseResidual < 2.0e-2
        && result.conditionEstimate < 1.0e8;
    return result;
}

VTEigensystem9 vtBlockEigensystem9(
    in float symbol[81],
    int axis
)
{
    VTEigensystem9 result = {};
    for (int index = 0; index < 81; ++index)
        result.symbol[index] = symbol[index];
    int longitudinal[5] = {
        0, 1, 2 + axis, 5, 6 + axis
    };
    float longitudinalSymbol[25] = {};
    for (int row = 0; row < 5; ++row)
    {
        for (int column = 0; column < 5; ++column)
        {
            longitudinalSymbol[row * 5 + column] =
                symbol[
                    longitudinal[row] * 9 + longitudinal[column]
                ];
        }
    }
    VTEigensystem longitudinalSystem = vtEigensystem(
        longitudinalSymbol
    );
    for (int mode = 0; mode < 5; ++mode)
    {
        result.values[mode] = longitudinalSystem.values[mode];
        for (int row = 0; row < 5; ++row)
        {
            result.right[
                longitudinal[row] * 9 + mode
            ] = longitudinalSystem.right[row * 5 + mode];
        }
    }

    int transverse[4] = {};
    int transverseCount = 0;
    for (int spatial = 0; spatial < 3; ++spatial)
    {
        if (spatial == axis)
            continue;
        transverse[transverseCount++] = 2 + spatial;
        transverse[transverseCount++] = 6 + spatial;
    }
    bool transverseReal = true;
    for (int pair = 0; pair < 2; ++pair)
    {
        int first = transverse[2 * pair];
        int second = transverse[2 * pair + 1];
        float a = symbol[first * 9 + first];
        float b = symbol[first * 9 + second];
        float c = symbol[second * 9 + first];
        float d = symbol[second * 9 + second];
        float rawDiscriminant =
            (a - d) * (a - d) + 4.0 * b * c;
        transverseReal =
            transverseReal && rawDiscriminant >= -1.0e-5;
        float root = sqrt(max(rawDiscriminant, 0.0));
        float pairValues[2] = {
            0.5 * (a + d - root),
            0.5 * (a + d + root)
        };
        for (int localMode = 0; localMode < 2; ++localMode)
        {
            int mode = 5 + 2 * pair + localMode;
            float value = pairValues[localMode];
            result.values[mode] = value;
            float firstValue = b;
            float secondValue = value - a;
            float normSquared =
                firstValue * firstValue + secondValue * secondValue;
            if (normSquared < 1.0e-18)
            {
                firstValue = value - d;
                secondValue = c;
                normSquared =
                    firstValue * firstValue
                    + secondValue * secondValue;
            }
            if (normSquared < 1.0e-18)
            {
                firstValue = localMode == 0 ? 1.0 : 0.0;
                secondValue = localMode == 0 ? 0.0 : 1.0;
                normSquared = 1.0;
            }
            float inverseNorm = rsqrt(normSquared);
            firstValue *= inverseNorm;
            secondValue *= inverseNorm;
            if (
                abs(firstValue) >= abs(secondValue)
                && firstValue < 0.0
            )
            {
                firstValue = -firstValue;
                secondValue = -secondValue;
            }
            else if (
                abs(secondValue) > abs(firstValue)
                && secondValue < 0.0
            )
            {
                firstValue = -firstValue;
                secondValue = -secondValue;
            }
            result.right[first * 9 + mode] = firstValue;
            result.right[second * 9 + mode] = secondValue;
        }
    }

    bool inverseValid = vtInvert9(result.right, result.left);
    result.maximumResidual = 0.0;
    result.inverseResidual = 0.0;
    result.maximumSpeed = 0.0;
    float rightNorm = 0.0;
    float leftNorm = 0.0;
    for (int mode = 0; mode < 9; ++mode)
    {
        result.maximumSpeed = max(
            result.maximumSpeed, abs(result.values[mode])
        );
        for (int row = 0; row < 9; ++row)
        {
            float applied = 0.0;
            for (int column = 0; column < 9; ++column)
            {
                applied +=
                    symbol[row * 9 + column]
                    * result.right[column * 9 + mode];
            }
            result.maximumResidual = max(
                result.maximumResidual,
                abs(
                    applied
                    - result.values[mode]
                    * result.right[row * 9 + mode]
                )
            );
        }
    }
    for (int row = 0; row < 9; ++row)
    {
        for (int column = 0; column < 9; ++column)
        {
            float product = 0.0;
            for (int inner = 0; inner < 9; ++inner)
            {
                product +=
                    result.left[row * 9 + inner]
                    * result.right[inner * 9 + column];
            }
            result.inverseResidual = max(
                result.inverseResidual,
                abs(product - (row == column ? 1.0 : 0.0))
            );
            rightNorm +=
                result.right[row * 9 + column]
                * result.right[row * 9 + column];
            leftNorm +=
                result.left[row * 9 + column]
                * result.left[row * 9 + column];
        }
    }
    result.conditionEstimate = sqrt(rightNorm * leftNorm);
    result.valid =
        longitudinalSystem.valid
        && transverseReal
        && inverseValid
        && isfinite(result.maximumResidual)
        && isfinite(result.maximumSpeed)
        && result.maximumResidual < 2.0e-2
        && result.inverseResidual < 2.0e-2
        && result.conditionEstimate < 1.0e8;
    return result;
}

VTEigensystem9 vtCarterSymbol9(
    in float x[9],
    in VTGeometry3 geometry,
    int axis,
    int modelMode,
    int phaseOverride
)
{
    float timeJacobian[81] = {};
    float fluxPathJacobian[81] = {};
    VTPrincipal9 base = vtPrincipal9(
        x, geometry, axis, modelMode, phaseOverride
    );
    for (int column = 0; column < 9; ++column)
    {
        float step =
            VT_CHARACTERISTIC_STEP9 * max(1.0, abs(x[column]));
        float plus[9] = {};
        float minus[9] = {};
        for (int index = 0; index < 9; ++index)
        {
            plus[index] = x[index];
            minus[index] = x[index];
        }
        plus[column] += step;
        minus[column] -= step;
        VTPrincipal9 plusPrincipal = vtPrincipal9(
            plus, geometry, axis, modelMode, phaseOverride
        );
        VTPrincipal9 minusPrincipal = vtPrincipal9(
            minus, geometry, axis, modelMode, phaseOverride
        );
        float inverseStep = 1.0 / (2.0 * step);
        for (int row = 0; row < 9; ++row)
        {
            timeJacobian[row * 9 + column] =
                (
                    plusPrincipal.conserved[row]
                    - minusPrincipal.conserved[row]
                )
                * inverseStep;
            fluxPathJacobian[row * 9 + column] =
                (
                    plusPrincipal.flux[row]
                    - minusPrincipal.flux[row]
                )
                * inverseStep;
        }
        float chiZeroDerivative =
            (plusPrincipal.chiZero - minusPrincipal.chiZero)
            * inverseStep;
        float pathDerivative = chiZeroDerivative;
        for (int component = 0; component < 3; ++component)
        {
            pathDerivative += base.carrierTransport[component]
                * (plusPrincipal.chiDown[component] - minusPrincipal.chiDown[component])
                * inverseStep;
        }
        fluxPathJacobian[(6 + axis) * 9 + column] -=
            base.carrierEulerian * pathDerivative;
    }
    float inverseTime[81] = {};
    float symbol[81] = {};
    if (!vtInvert9(timeJacobian, inverseTime))
        return vtEigensystem9(symbol);
    vtMatMul9(fluxPathJacobian, inverseTime, symbol);
    return vtEigensystem9(symbol);
}

void vtCarterPathSymbol9(
    in float x[9],
    in VTGeometry3 geometry,
    int axis,
    int modelMode,
    int phaseOverride,
    out float pathSymbol[81]
)
{
    float timeJacobian[81] = {};
    float pathPrimitive[81] = {};
    VTPrincipal9 base = vtPrincipal9(
        x, geometry, axis, modelMode, phaseOverride
    );
    for (int column = 0; column < 9; ++column)
    {
        float step =
            VT_CHARACTERISTIC_STEP9 * max(1.0, abs(x[column]));
        float plus[9] = {};
        float minus[9] = {};
        for (int index = 0; index < 9; ++index)
        {
            plus[index] = x[index];
            minus[index] = x[index];
        }
        plus[column] += step;
        minus[column] -= step;
        VTPrincipal9 plusPrincipal = vtPrincipal9(
            plus, geometry, axis, modelMode, phaseOverride
        );
        VTPrincipal9 minusPrincipal = vtPrincipal9(
            minus, geometry, axis, modelMode, phaseOverride
        );
        float inverseStep = 1.0 / (2.0 * step);
        for (int row = 0; row < 9; ++row)
        {
            timeJacobian[row * 9 + column] =
                (
                    plusPrincipal.conserved[row]
                    - minusPrincipal.conserved[row]
                )
                * inverseStep;
        }
        float chiZeroDerivative =
            (plusPrincipal.chiZero - minusPrincipal.chiZero)
            * inverseStep;
        float pathDerivative = chiZeroDerivative;
        for (int component = 0; component < 3; ++component)
        {
            pathDerivative += base.carrierTransport[component]
                * (plusPrincipal.chiDown[component] - minusPrincipal.chiDown[component])
                * inverseStep;
        }
        pathPrimitive[(6 + axis) * 9 + column] =
            -base.carrierEulerian * pathDerivative;
    }
    float inverseTime[81] = {};
    if (!vtInvert9(timeJacobian, inverseTime))
    {
        for (int index = 0; index < 81; ++index)
            pathSymbol[index] = 0.0;
        return;
    }
    vtMatMul9(pathPrimitive, inverseTime, pathSymbol);
}

void theory33FullEigensystemKernel(
    int point,
    int axis,
    int modelMode,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> symbol,
    IWTensor<float, 2> rightBasis,
    IWTensor<float, 2> leftBasis,
    IWTensor<float, 2> eigenvalues,
    IWTensor<float, 2> audit
)
{
    float x[9] = {};
    float geometryValues[13] = {};
    for (int index = 0; index < 9; ++index)
        x[index] = primitive[point, index];
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    int phaseOverride = -1;
    if (modelMode == 1)
    {
        VTPrincipal9 classified = vtPrincipal9(
            x, geometry, clamp(axis, 0, 2), modelMode, -1
        );
        phaseOverride = classified.phaseTwo ? 1 : 0;
    }
    VTEigensystem9 system = vtCarterSymbol9(
        x,
        geometry,
        clamp(axis, 0, 2),
        modelMode,
        phaseOverride
    );
    for (int index = 0; index < 81; ++index)
    {
        symbol[point, index] = system.symbol[index];
        rightBasis[point, index] = system.right[index];
        leftBasis[point, index] = system.left[index];
    }
    for (int mode = 0; mode < 9; ++mode)
        eigenvalues[point, mode] = system.values[mode];
    audit[point, 0] = system.maximumSpeed;
    audit[point, 1] = system.maximumResidual;
    audit[point, 2] = system.inverseResidual;
    audit[point, 3] = system.conditionEstimate;
    audit[point, 4] = system.valid ? 1.0 : 0.0;
    audit[point, 5] = geometry.valid ? 1.0 : 0.0;
}

void theory33GeometryAuditKernel(
    int point,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> audit
)
{
    float values[13] = {};
    for (int index = 0; index < 13; ++index)
        values[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(values);
    for (int index = 0; index < 9; ++index)
    {
        audit[point, index] = geometry.metric[index];
        audit[point, 9 + index] = geometry.inverseMetric[index];
    }
    for (int row = 0; row < 3; ++row)
    {
        float lowered = 0.0;
        for (int column = 0; column < 3; ++column)
        {
            lowered +=
                geometry.metric[row * 3 + column]
                * primitive[point, 2 + column];
        }
        audit[point, 18 + row] = lowered;
    }
    audit[point, 21] = geometry.sqrtDeterminant;
}

void ccz4ProjectGeometryKernel(
    int point,
    ITensor<float, 2> stateInput,
    IWTensor<float, 2> stateOutput,
    IWTensor<float, 2> audit
)
{
    float metricValues[13] = {};
    for (int index = 0; index < 9; ++index)
    {
        metricValues[index] = 0.5 * (
            stateInput[point, index]
            + stateInput[point, (index % 3) * 3 + index / 3]
        );
    }
    metricValues[9] = 1.0;
    VTGeometry3 initial = vtGeometry3(metricValues);
    float rescale = pow(
        max(
            initial.sqrtDeterminant * initial.sqrtDeterminant,
            1.0e-20
        ),
        -1.0 / 3.0
    );
    for (int index = 0; index < 9; ++index)
        metricValues[index] *= rescale;
    VTGeometry3 projected = vtGeometry3(metricValues);
    float conformalA[9] = {};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            conformalA[row * 3 + column] = 0.5 * (
                stateInput[point, 9 + row * 3 + column]
                + stateInput[point, 9 + column * 3 + row]
            );
        }
    }
    float traceA = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            traceA +=
                projected.inverseMetric[row * 3 + column]
                * conformalA[row * 3 + column];
        }
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            stateOutput[point, row * 3 + column] =
                metricValues[row * 3 + column];
            stateOutput[point, 9 + row * 3 + column] =
                conformalA[row * 3 + column]
                - metricValues[row * 3 + column] * traceA / 3.0;
        }
    }
    stateOutput[point, 18] =
        max(stateInput[point, 18], 1.0e-8);
    for (int index = 19; index < 24; ++index)
        stateOutput[point, index] = stateInput[point, index];
    stateOutput[point, 24] =
        max(stateInput[point, 24], 1.0e-8);
    for (int index = 25; index < 31; ++index)
        stateOutput[point, index] = stateInput[point, index];
    float projectedTrace = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            projectedTrace +=
                projected.inverseMetric[row * 3 + column]
                * (
                    conformalA[row * 3 + column]
                    - metricValues[row * 3 + column] * traceA / 3.0
                );
        }
    }
    audit[point, 0] =
        abs(
            projected.sqrtDeterminant
            * projected.sqrtDeterminant
            - 1.0
        );
    audit[point, 1] = abs(projectedTrace);
    audit[point, 2] = projected.valid ? 1.0 : 0.0;
}

void ccz4PhysicalGeometryKernel(
    int point,
    ITensor<float, 2> state,
    IWTensor<float, 2> geometryOutput,
    IWTensor<float, 2> extrinsicCurvature
)
{
    float phi = max(state[point, 18], 1.0e-8);
    float inversePhiSquared = 1.0 / (phi * phi);
    float traceK = state[point, 19];
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            float conformalMetric =
                state[point, row * 3 + column];
            geometryOutput[point, row * 3 + column] =
                conformalMetric * inversePhiSquared;
            extrinsicCurvature[point, row * 3 + column] =
                (
                    state[point, 9 + row * 3 + column]
                    + traceK * conformalMetric / 3.0
                )
                * inversePhiSquared;
        }
    }
    geometryOutput[point, 9] = state[point, 24];
    geometryOutput[point, 10] = state[point, 25];
    geometryOutput[point, 11] = state[point, 26];
    geometryOutput[point, 12] = state[point, 27];
}

void ccz4MatterSourceKernel(
    int point,
    float kappaGravity,
    ITensor<float, 2> projectedState,
    ITensor<float, 2> matterStress,
    IWTensor<float, 2> source
)
{
    for (int index = 0; index < 31; ++index)
        source[point, index] = 0.0;
    float metricValues[13] = {};
    for (int index = 0; index < 9; ++index)
        metricValues[index] = projectedState[point, index];
    metricValues[9] = 1.0;
    VTGeometry3 conformal = vtGeometry3(metricValues);
    float phi = max(projectedState[point, 18], 1.0e-8);
    float alpha = projectedState[point, 24];
    float physicalTrace = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            physicalTrace +=
                phi
                * phi
                * conformal.inverseMetric[row * 3 + column]
                * matterStress[point, 4 + row * 3 + column];
        }
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            source[point, 9 + row * 3 + column] =
                -kappaGravity
                * alpha
                * phi
                * phi
                * matterStress[point, 4 + row * 3 + column]
                + kappaGravity
                * alpha
                * projectedState[point, row * 3 + column]
                * physicalTrace
                / 3.0;
        }
    }
    float rho = matterStress[point, 0];
    source[point, 19] =
        0.5 * kappaGravity * alpha * (physicalTrace - 3.0 * rho);
    source[point, 20] = -kappaGravity * alpha * rho;
    for (int row = 0; row < 3; ++row)
    {
        float momentumUpConformal = 0.0;
        for (int column = 0; column < 3; ++column)
        {
            momentumUpConformal +=
                conformal.inverseMetric[row * 3 + column]
                * matterStress[point, 1 + column];
        }
        source[point, 21 + row] =
            -2.0 * kappaGravity * alpha * momentumUpConformal;
    }
}

void ccz4SumRhsKernel(
    int point,
    ITensor<float, 2> vacuumRhs,
    ITensor<float, 2> matterRhs,
    IWTensor<float, 2> rhs
)
{
    for (int component = 0; component < 31; ++component)
    {
        rhs[point, component] =
            vacuumRhs[point, component] + matterRhs[point, component];
    }
}

void ccz4StageKernel(
    int point,
    float dt,
    float baseWeight,
    float evolvedWeight,
    ITensor<float, 2> baseState,
    ITensor<float, 2> stageState,
    ITensor<float, 2> rhs,
    IWTensor<float, 2> output
)
{
    for (int component = 0; component < 31; ++component)
    {
        output[point, component] =
            baseWeight * baseState[point, component]
            + evolvedWeight
            * (stageState[point, component] + dt * rhs[point, component]);
    }
}

int vtGridNeighbor(
    int point,
    int axis,
    int offset,
    int nx,
    int ny,
    int nz
)
{
    int x = point % nx;
    int y = (point / nx) % ny;
    int z = point / (nx * ny);
    // Do not use a negative integer remainder here.  It is lowered
    // inconsistently by some Windows Vulkan compiler paths.  The largest
    // resident stencil offset is three cells; explicit sign-free corrections
    // preserve periodic wrapping for every supported active-axis size.
    if (axis == 0)
    {
        x += offset;
        if (x < 0) x += nx;
        if (x < 0) x += nx;
        if (x < 0) x += nx;
        if (x < 0) x += nx;
        if (x >= nx) x -= nx;
        if (x >= nx) x -= nx;
        if (x >= nx) x -= nx;
        if (x >= nx) x -= nx;
    }
    else if (axis == 1)
    {
        y += offset;
        if (y < 0) y += ny;
        if (y < 0) y += ny;
        if (y < 0) y += ny;
        if (y < 0) y += ny;
        if (y >= ny) y -= ny;
        if (y >= ny) y -= ny;
        if (y >= ny) y -= ny;
        if (y >= ny) y -= ny;
    }
    else
    {
        z += offset;
        if (z < 0) z += nz;
        if (z < 0) z += nz;
        if (z < 0) z += nz;
        if (z < 0) z += nz;
        if (z >= nz) z -= nz;
        if (z >= nz) z -= nz;
        if (z >= nz) z -= nz;
        if (z >= nz) z -= nz;
    }
    return x + nx * (y + ny * z);
}

float vtMixedDerivative(
    int point,
    int component,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> field
)
{
    int count = axis == 0 ? nx : (axis == 1 ? ny : nz);
    if (count <= 1)
        return 0.0;
    int plus = vtGridNeighbor(point, axis, 1, nx, ny, nz);
    int minus = vtGridNeighbor(point, axis, -1, nx, ny, nz);
    float inverse = axis == 0
        ? inverseDx
        : (axis == 1 ? inverseDy : inverseDz);
    return 0.5 * inverse * (
        field[plus, component] - field[minus, component]
    );
}

float vtMixedPhi(
    int point,
    int fieldIndex,
    ITensor<float, 2> dynamicState,
    ITensor<float, 2> matterState,
    ITensor<float, 2> geometryInput
)
{
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float rightA =
        dynamicState[point, 12] / geometry.sqrtDeterminant;
    float rightB =
        dynamicState[point, 13] / geometry.sqrtDeterminant
        + matterState[point, 1] / geometry.sqrtDeterminant;
    // Inverse of the default Theory 3 Gauss matrix
    // [[m_A^2+mu^2, -mu^2],[-mu^2,m_B^2+mu^2]]
    // with m_A=m_B=0.5 and mu^2=0.25.
    return fieldIndex == 0
        ? (8.0 * rightA + 4.0 * rightB) / 3.0
        : (4.0 * rightA + 8.0 * rightB) / 3.0;
}

float vtMixedFieldStrength(
    int point,
    int fieldIndex,
    int row,
    int column,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> dynamicState
)
{
    int offset = fieldIndex == 0 ? 0 : 6;
    return
        vtMixedDerivative(
            point, offset + column, row,
            nx, ny, nz, inverseDx, inverseDy, inverseDz,
            dynamicState
        )
        - vtMixedDerivative(
            point, offset + row, column,
            nx, ny, nz, inverseDx, inverseDy, inverseDz,
            dynamicState
        );
}

float vtMixedMomentumFlux(
    int point,
    int fieldIndex,
    int derivativeAxis,
    int component,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> dynamicState,
    ITensor<float, 2> geometryInput
)
{
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float raised = 0.0;
    for (int lowerRow = 0; lowerRow < 3; ++lowerRow)
    {
        for (int lowerColumn = 0; lowerColumn < 3; ++lowerColumn)
        {
            raised +=
                geometry.inverseMetric[
                    derivativeAxis * 3 + lowerRow
                ]
                * geometry.inverseMetric[
                    component * 3 + lowerColumn
                ]
                * vtMixedFieldStrength(
                    point, fieldIndex, lowerRow, lowerColumn,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    dynamicState
                );
        }
    }
    return geometry.lapse * geometry.sqrtDeterminant * raised;
}

float vtMixedCurrent(
    int point,
    int fieldIndex,
    int axis,
    ITensor<float, 2> dynamicState,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput
)
{
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float aUp = 0.0;
    float bUp = 0.0;
    for (int column = 0; column < 3; ++column)
    {
        aUp +=
            geometry.inverseMetric[axis * 3 + column]
            * dynamicState[point, column];
        bUp +=
            geometry.inverseMetric[axis * 3 + column]
            * dynamicState[point, 6 + column];
    }
    if (fieldIndex == 0)
        return 0.5 * aUp - 0.25 * bUp;
    float densityD = exp(clamp(primitive[point, 5], -20.0, 4.0));
    float targetCurrent = densityD * primitive[point, 6 + axis];
    return -0.25 * aUp + 0.5 * bUp - targetCurrent;
}

void theory33MixedVectorInputKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> dynamicState,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> vectorInput
)
{
    for (int component = 0; component < 14; ++component)
        vectorInput[point, component] = dynamicState[point, component];
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    for (int fieldIndex = 0; fieldIndex < 2; ++fieldIndex)
    {
        int output = 14 + 3 * fieldIndex;
        vectorInput[point, output] =
            vtMixedFieldStrength(
                point, fieldIndex, 1, 2,
                nx, ny, nz, inverseDx, inverseDy, inverseDz,
                dynamicState
            )
            / geometry.sqrtDeterminant;
        vectorInput[point, output + 1] =
            vtMixedFieldStrength(
                point, fieldIndex, 2, 0,
                nx, ny, nz, inverseDx, inverseDy, inverseDz,
                dynamicState
            )
            / geometry.sqrtDeterminant;
        vectorInput[point, output + 2] =
            vtMixedFieldStrength(
                point, fieldIndex, 0, 1,
                nx, ny, nz, inverseDx, inverseDy, inverseDz,
                dynamicState
            )
            / geometry.sqrtDeterminant;
    }
}

void theory33MixedVectorRhsKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float cleaningDamping,
    ITensor<float, 2> dynamicState,
    ITensor<float, 2> matterState,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> rhs
)
{
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float inverseSpacing[3] = {inverseDx, inverseDy, inverseDz};
    int axisCount[3] = {nx, ny, nz};
    float shiftDivergence = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (axisCount[axis] > 1)
        {
            shiftDivergence += vtMixedDerivative(
                point, 10 + axis, axis,
                nx, ny, nz, inverseDx, inverseDy, inverseDz,
                geometryInput
            );
        }
    }
    float aUp[3] = {};
    float bUp[3] = {};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            aUp[row] +=
                geometry.inverseMetric[row * 3 + column]
                * dynamicState[point, column];
            bUp[row] +=
                geometry.inverseMetric[row * 3 + column]
                * dynamicState[point, 6 + column];
        }
    }
    for (int fieldIndex = 0; fieldIndex < 2; ++fieldIndex)
    {
        int potentialOffset = fieldIndex == 0 ? 0 : 6;
        int momentumOffset = fieldIndex == 0 ? 3 : 9;
        int longitudinalIndex = 12 + fieldIndex;
        int cleaningIndex = 14 + fieldIndex;
        float phi = vtMixedPhi(
            point, fieldIndex, dynamicState, matterState, geometryInput
        );
        for (int component = 0; component < 3; ++component)
        {
            float piLower = 0.0;
            for (int column = 0; column < 3; ++column)
            {
                piLower +=
                    geometry.metric[component * 3 + column]
                    * dynamicState[point, momentumOffset + column];
            }
            float potentialRhs =
                geometry.lapse * piLower / geometry.sqrtDeterminant;
            float momentumRhs = 0.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (axisCount[axis] <= 1)
                    continue;
                int plus = vtGridNeighbor(
                    point, axis, 1, nx, ny, nz
                );
                int minus = vtGridNeighbor(
                    point, axis, -1, nx, ny, nz
                );
                float plusPhi = vtMixedPhi(
                    plus, fieldIndex,
                    dynamicState, matterState, geometryInput
                );
                float minusPhi = vtMixedPhi(
                    minus, fieldIndex,
                    dynamicState, matterState, geometryInput
                );
                potentialRhs -=
                    component == axis
                    ? 0.5 * inverseSpacing[axis]
                    * (
                        geometryInput[plus, 9] * plusPhi
                        - geometryInput[minus, 9] * minusPhi
                    )
                    : 0.0;
                potentialRhs +=
                    geometry.shift[axis]
                    * vtMixedDerivative(
                        point, potentialOffset + component, axis,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        dynamicState
                    );
                potentialRhs +=
                    dynamicState[point, potentialOffset + axis]
                    * vtMixedDerivative(
                        point, 10 + axis, component,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        geometryInput
                    );
                float plusFlux = vtMixedMomentumFlux(
                    plus, fieldIndex, axis, component,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    dynamicState, geometryInput
                );
                float minusFlux = vtMixedMomentumFlux(
                    minus, fieldIndex, axis, component,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    dynamicState, geometryInput
                );
                momentumRhs +=
                    0.5 * inverseSpacing[axis]
                    * (plusFlux - minusFlux);
                momentumRhs +=
                    geometry.shift[axis]
                    * vtMixedDerivative(
                        point, momentumOffset + component, axis,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        dynamicState
                    );
                momentumRhs -=
                    dynamicState[point, momentumOffset + axis]
                    * vtMixedDerivative(
                        point, 10 + component, axis,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        geometryInput
                    );
            }
            momentumRhs +=
                dynamicState[point, momentumOffset + component]
                * shiftDivergence;
            float cleaningGradient = 0.0;
            if (axisCount[component] > 1)
            {
                int plus = vtGridNeighbor(
                    point, component, 1, nx, ny, nz
                );
                int minus = vtGridNeighbor(
                    point, component, -1, nx, ny, nz
                );
                cleaningGradient =
                    0.5 * inverseSpacing[component]
                    * (
                        geometryInput[plus, 9]
                        * dynamicState[plus, cleaningIndex]
                        - geometryInput[minus, 9]
                        * dynamicState[minus, cleaningIndex]
                    );
            }
            float raisedCleaning = 0.0;
            for (int column = 0; column < 3; ++column)
            {
                if (axisCount[column] <= 1)
                    continue;
                int plus = vtGridNeighbor(
                    point, column, 1, nx, ny, nz
                );
                int minus = vtGridNeighbor(
                    point, column, -1, nx, ny, nz
                );
                float gradient =
                    0.5 * inverseSpacing[column]
                    * (
                        geometryInput[plus, 9]
                        * dynamicState[plus, cleaningIndex]
                        - geometryInput[minus, 9]
                        * dynamicState[minus, cleaningIndex]
                    );
                raisedCleaning +=
                    geometry.inverseMetric[component * 3 + column]
                    * gradient;
            }
            momentumRhs += geometry.sqrtDeterminant * raisedCleaning;
            float mixing = aUp[component] - bUp[component];
            if (fieldIndex == 0)
            {
                momentumRhs -=
                    geometry.lapse * geometry.sqrtDeterminant
                    * (0.25 * aUp[component] + 0.25 * mixing);
            }
            else
            {
                float densityD = exp(
                    clamp(primitive[point, 5], -20.0, 4.0)
                );
                float targetCurrent =
                    densityD * primitive[point, 6 + component];
                momentumRhs -=
                    geometry.lapse * geometry.sqrtDeterminant
                    * (
                        0.25 * bUp[component]
                        - 0.25 * mixing
                        - targetCurrent
                    );
            }
            rhs[point, potentialOffset + component] = potentialRhs;
            rhs[point, momentumOffset + component] = momentumRhs;
        }
        float charge =
            dynamicState[point, longitudinalIndex]
            / geometry.sqrtDeterminant;
        float longitudinalRhs = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            if (axisCount[axis] <= 1)
                continue;
            int plus = vtGridNeighbor(
                point, axis, 1, nx, ny, nz
            );
            int minus = vtGridNeighbor(
                point, axis, -1, nx, ny, nz
            );
            float plusGeometryValues[13] = {};
            float minusGeometryValues[13] = {};
            for (int index = 0; index < 13; ++index)
            {
                plusGeometryValues[index] =
                    geometryInput[plus, index];
                minusGeometryValues[index] =
                    geometryInput[minus, index];
            }
            VTGeometry3 plusGeometry = vtGeometry3(
                plusGeometryValues
            );
            VTGeometry3 minusGeometry = vtGeometry3(
                minusGeometryValues
            );
            float plusCharge =
                dynamicState[plus, longitudinalIndex]
                / plusGeometry.sqrtDeterminant;
            float minusCharge =
                dynamicState[minus, longitudinalIndex]
                / minusGeometry.sqrtDeterminant;
            float plusFlux =
                plusGeometry.sqrtDeterminant
                * (
                    plusGeometry.lapse
                    * vtMixedCurrent(
                        plus, fieldIndex, axis,
                        dynamicState, primitive, geometryInput
                    )
                    - plusGeometry.shift[axis] * plusCharge
                );
            float minusFlux =
                minusGeometry.sqrtDeterminant
                * (
                    minusGeometry.lapse
                    * vtMixedCurrent(
                        minus, fieldIndex, axis,
                        dynamicState, primitive, geometryInput
                    )
                    - minusGeometry.shift[axis] * minusCharge
                );
            longitudinalRhs -=
                0.5 * inverseSpacing[axis]
                * (plusFlux - minusFlux);
        }
        rhs[point, longitudinalIndex] = longitudinalRhs;
        float divergencePi = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            divergencePi += vtMixedDerivative(
                point, momentumOffset + axis, axis,
                nx, ny, nz,
                inverseDx, inverseDy, inverseDz,
                dynamicState
            );
        }
        float constraint =
            -divergencePi / geometry.sqrtDeterminant + charge;
        float cleaningRhs =
            -geometry.lapse
            * (
                constraint
                + cleaningDamping
                * dynamicState[point, cleaningIndex]
            );
        for (int axis = 0; axis < 3; ++axis)
        {
            cleaningRhs +=
                geometry.shift[axis]
                * vtMixedDerivative(
                    point, cleaningIndex, axis,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    dynamicState
                );
        }
        rhs[point, cleaningIndex] = cleaningRhs;
    }
}

void theory33MixedVectorStageKernel(
    int point,
    float dt,
    float baseWeight,
    float evolvedWeight,
    ITensor<float, 2> baseState,
    ITensor<float, 2> stageState,
    ITensor<float, 2> rhs,
    IWTensor<float, 2> output
)
{
    for (int component = 0; component < 16; ++component)
    {
        output[point, component] =
            baseWeight * baseState[point, component]
            + evolvedWeight
            * (stageState[point, component] + dt * rhs[point, component]);
    }
}

// -------------------------------------------------------------------------
// Periodic CMC/Picard initial data
//
// The elliptic state shares the evolution ABI's float32 precision. Global
// reductions are deterministic; qualification records the resulting
// residual floor separately from the float64 CPU oracle.
// Columns are psi, W^x, W^y, W^z.

float vtCMCInverseSpacing(
    int axis,
    float inverseDx,
    float inverseDy,
    float inverseDz
)
{
    return axis == 0
        ? inverseDx
        : (axis == 1 ? inverseDy : inverseDz);
}

int vtCMCAxisCount(int axis, int nx, int ny, int nz)
{
    return axis == 0 ? nx : (axis == 1 ? ny : nz);
}

float vtCMCDerivative(
    int point,
    int component,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> state
)
{
    if (vtCMCAxisCount(axis, nx, ny, nz) <= 1)
        return 0.0;
    int plus = vtGridNeighbor(point, axis, 1, nx, ny, nz);
    int minus = vtGridNeighbor(point, axis, -1, nx, ny, nz);
    return 0.5
        * vtCMCInverseSpacing(
            axis, inverseDx, inverseDy, inverseDz
        )
        * (state[plus, component] - state[minus, component]);
}

float vtCMCSecondDerivative(
    int point,
    int component,
    int firstAxis,
    int secondAxis,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> state
)
{
    if (
        vtCMCAxisCount(firstAxis, nx, ny, nz) <= 1
        || vtCMCAxisCount(secondAxis, nx, ny, nz) <= 1
    )
        return 0.0;
    float firstInverse = vtCMCInverseSpacing(
        firstAxis, inverseDx, inverseDy, inverseDz
    );
    float secondInverse = vtCMCInverseSpacing(
        secondAxis, inverseDx, inverseDy, inverseDz
    );
    if (firstAxis == secondAxis)
    {
        int plus = vtGridNeighbor(
            point, firstAxis, 2, nx, ny, nz
        );
        int minus = vtGridNeighbor(
            point, firstAxis, -2, nx, ny, nz
        );
        return 0.25
            * firstInverse
            * firstInverse
            * (
                state[plus, component]
                - 2.0 * state[point, component]
                + state[minus, component]
            );
    }
    int plusPlus = vtGridNeighbor(
        vtGridNeighbor(point, firstAxis, 1, nx, ny, nz),
        secondAxis, 1, nx, ny, nz
    );
    int plusMinus = vtGridNeighbor(
        vtGridNeighbor(point, firstAxis, 1, nx, ny, nz),
        secondAxis, -1, nx, ny, nz
    );
    int minusPlus = vtGridNeighbor(
        vtGridNeighbor(point, firstAxis, -1, nx, ny, nz),
        secondAxis, 1, nx, ny, nz
    );
    int minusMinus = vtGridNeighbor(
        vtGridNeighbor(point, firstAxis, -1, nx, ny, nz),
        secondAxis, -1, nx, ny, nz
    );
    return 0.25
        * firstInverse
        * secondInverse
        * (
            state[plusPlus, component]
            - state[plusMinus, component]
            - state[minusPlus, component]
            + state[minusMinus, component]
        );
}

float vtCMCLaplacian(
    int point,
    int component,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> state
)
{
    float result = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (vtCMCAxisCount(axis, nx, ny, nz) <= 1)
            continue;
        int plus = vtGridNeighbor(point, axis, 1, nx, ny, nz);
        int minus = vtGridNeighbor(point, axis, -1, nx, ny, nz);
        float inverse = vtCMCInverseSpacing(
            axis, inverseDx, inverseDy, inverseDz
        );
        result += inverse * inverse * (
            state[plus, component]
            - 2.0 * state[point, component]
            + state[minus, component]
        );
    }
    return result;
}

void vtCMCAbar(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> state,
    out float Abar[9]
)
{
    float divergence = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        divergence += vtCMCDerivative(
            point, 1 + axis, axis,
            nx, ny, nz, inverseDx, inverseDy, inverseDz,
            state
        );
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            Abar[row * 3 + column] =
                vtCMCDerivative(
                    point, 1 + row, column,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    state
                )
                + vtCMCDerivative(
                    point, 1 + column, row,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    state
                )
                - (row == column
                    ? (2.0 / 3.0) * divergence
                    : 0.0);
        }
    }
}

void vtCMCResidual(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float kappaGravity,
    ITensor<float, 2> state,
    ITensor<float, 2> matterStress,
    ITensor<float, 2> control,
    out float hamiltonian,
    out float momentum[3]
)
{
    float Abar[9] = {};
    vtCMCAbar(
        point, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        state, Abar
    );
    float AbarSquared = 0.0;
    for (int component = 0; component < 9; ++component)
        AbarSquared += Abar[component] * Abar[component];
    float psi = max(state[point, 0], 1.0e-12);
    float psi5 = pow(psi, 5.0);
    float traceKSquared = control[0, 0];
    hamiltonian =
        vtCMCLaplacian(
            point, 0, nx, ny, nz,
            inverseDx, inverseDy, inverseDz, state
        )
        + 0.125 * AbarSquared * pow(psi, -7.0)
        - (traceKSquared / 12.0) * psi5
        + 0.25
        * kappaGravity
        * float(matterStress[point, 0])
        * psi5;
    for (int row = 0; row < 3; ++row)
    {
        float vectorLaplacian = vtCMCLaplacian(
            point, 1 + row, nx, ny, nz,
            inverseDx, inverseDy, inverseDz, state
        );
        float gradientDivergence = 0.0;
        for (int inner = 0; inner < 3; ++inner)
        {
            gradientDivergence += vtCMCSecondDerivative(
                point, 1 + inner, row, inner,
                nx, ny, nz,
                inverseDx, inverseDy, inverseDz,
                state
            );
        }
        momentum[row] =
            vectorLaplacian
            + gradientDivergence / 3.0
            - kappaGravity
            * pow(psi, 6.0)
            * float(matterStress[point, 1 + row]);
    }
}

void theory33CMCTraceKernel(
    int point,
    int numberPoints,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float kappaGravity,
    int expanding,
    ITensor<float, 2> state,
    ITensor<float, 2> matterStress,
    IWTensor<float, 2> control
)
{
    float Aterm = 0.0;
    float rhoTerm = 0.0;
    float denominator = 0.0;
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        float Abar[9] = {};
        vtCMCAbar(
            cell, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            state, Abar
        );
        float AbarSquared = 0.0;
        for (int component = 0; component < 9; ++component)
            AbarSquared += Abar[component] * Abar[component];
        float psi = max(state[cell, 0], 1.0e-12);
        float psi5 = pow(psi, 5.0);
        Aterm += AbarSquared * pow(psi, -7.0);
        rhoTerm += float(matterStress[cell, 0]) * psi5;
        denominator += psi5;
    }
    float traceKSquared = (
        1.5 * Aterm + 3.0 * kappaGravity * rhoTerm
    ) / max(denominator, 1.0e-30);
    control[0, 0] = max(traceKSquared, 0.0);
    control[0, 1] =
        (expanding != 0 ? -1.0 : 1.0)
        * sqrt(max(traceKSquared, 0.0));
}

void theory33CMCUpdateKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float kappaGravity,
    float pseudoTimeStep,
    float minimumConformalFactor,
    float maximumConformalFactor,
    ITensor<float, 2> state,
    ITensor<float, 2> matterStress,
    ITensor<float, 2> control,
    IWTensor<float, 2> candidate
)
{
    float hamiltonian;
    float momentum[3] = {};
    vtCMCResidual(
        point, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        kappaGravity, state, matterStress, control,
        hamiltonian, momentum
    );
    candidate[point, 0] = clamp(
        state[point, 0] + pseudoTimeStep * hamiltonian,
        minimumConformalFactor,
        maximumConformalFactor
    );
    for (int axis = 0; axis < 3; ++axis)
    {
        candidate[point, 1 + axis] =
            state[point, 1 + axis]
            + pseudoTimeStep * momentum[axis];
    }
}

void theory33CMCMeanKernel(
    int point,
    int numberPoints,
    ITensor<float, 2> candidate,
    IWTensor<float, 2> control
)
{
    float sums[4] = {};
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        for (int component = 0; component < 4; ++component)
            sums[component] += candidate[cell, component];
    }
    for (int component = 0; component < 4; ++component)
        control[0, 2 + component] = sums[component] / numberPoints;
}

void theory33CMCNormalizeKernel(
    int point,
    ITensor<float, 2> candidate,
    ITensor<float, 2> control,
    IWTensor<float, 2> state
)
{
    state[point, 0] =
        candidate[point, 0] / max(control[0, 2], 1.0e-30);
    for (int axis = 0; axis < 3; ++axis)
    {
        state[point, 1 + axis] =
            candidate[point, 1 + axis] - control[0, 3 + axis];
    }
}

void theory33CMCResidualKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float kappaGravity,
    ITensor<float, 2> state,
    ITensor<float, 2> matterStress,
    ITensor<float, 2> control,
    IWTensor<float, 2> residual
)
{
    float hamiltonian;
    float momentum[3] = {};
    vtCMCResidual(
        point, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        kappaGravity, state, matterStress, control,
        hamiltonian, momentum
    );
    residual[point, 0] = hamiltonian * hamiltonian;
    residual[point, 1] =
        momentum[0] * momentum[0]
        + momentum[1] * momentum[1]
        + momentum[2] * momentum[2];
}

void theory33CMCReduceResidualKernel(
    int point,
    int numberPoints,
    ITensor<float, 2> state,
    ITensor<float, 2> residual,
    IWTensor<float, 2> control
)
{
    float hamiltonianSquared = 0.0;
    float momentumSquared = 0.0;
    float minimumPsi = 1.0e30;
    float maximumPsi = -1.0e30;
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        hamiltonianSquared += residual[cell, 0];
        momentumSquared += residual[cell, 1];
        minimumPsi = min(minimumPsi, state[cell, 0]);
        maximumPsi = max(maximumPsi, state[cell, 0]);
    }
    control[0, 6] = sqrt(hamiltonianSquared / numberPoints);
    control[0, 7] = sqrt(momentumSquared / numberPoints);
    control[0, 8] = minimumPsi;
    control[0, 9] = maximumPsi;
}

void theory33CMCPackCCZ4Kernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    ITensor<float, 2> state,
    ITensor<float, 2> control,
    IWTensor<float, 2> ccz4State
)
{
    for (int component = 0; component < 31; ++component)
        ccz4State[point, component] = 0.0;
    for (int axis = 0; axis < 3; ++axis)
        ccz4State[point, axis * 3 + axis] = 1.0;
    float Abar[9] = {};
    vtCMCAbar(
        point, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        state, Abar
    );
    float psi = max(state[point, 0], 1.0e-12);
    float scaleA = pow(psi, -6.0);
    for (int component = 0; component < 9; ++component)
        ccz4State[point, 9 + component] =
            float(scaleA * Abar[component]);
    ccz4State[point, 18] = float(pow(psi, -2.0));
    ccz4State[point, 19] = float(control[0, 1]);
    ccz4State[point, 24] = 1.0;
}

void theory33BalancedCounterflowKernel(
    int point,
    int nx,
    int ny,
    int nz,
    int modelMode,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> primitive,
    IWTensor<float, 2> audit
)
{
    int coordinates[3] = {
        point % nx,
        (point / nx) % ny,
        point / (nx * ny)
    };
    int counts[3] = { nx, ny, nz };
    int dimensions = (nx > 1 ? 1 : 0)
        + (ny > 1 ? 1 : 0)
        + (nz > 1 ? 1 : 0);
    float phases[3] = {};
    float modulation = 0.0;
    float sineProduct = 1.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (counts[axis] <= 1)
            continue;
        phases[axis] =
            6.283185307179586
            * float(coordinates[axis])
            / float(counts[axis]);
        modulation += cos(phases[axis]);
        sineProduct *= sin(phases[axis]);
    }
    modulation /= max(float(dimensions), 1.0);
    if (dimensions > 1)
        modulation += 0.25 * sineProduct;
    float density = 0.1 * (1.0 + 0.01 * modulation);
    float carrierDensity = 0.005 * (1.0 + 0.01 * modulation);
    float internalEnergy = 0.15;
    float entropy = log(max(
        (VT_GAMMA_AD - 1.0)
        * density
        * internalEnergy
        / pow(density, VT_GAMMA_AD),
        VT_EPSILON
    )) / (VT_GAMMA_AD - 1.0);
    float carrierVelocity[3] = {};
    carrierVelocity[0] = 0.125 + 0.001 * cos(phases[0]);
    if (ny > 1)
    {
        carrierVelocity[1] =
            0.025 * sin(phases[0]) + 0.01 * cos(phases[1]);
    }
    if (nz > 1)
    {
        carrierVelocity[2] =
            0.025 * sin(phases[1]) + 0.01 * cos(phases[2]);
    }
    float geometryValues[13] = {};
    for (int component = 0; component < 13; ++component)
        geometryValues[component] = geometryInput[point, component];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float carrierSpeedSquared = vtMetricDot3(
        carrierVelocity, geometry.metric, carrierVelocity
    );
    float carrierW = 1.0 / sqrt(max(1.0 - carrierSpeedSquared, 1.0e-8));
    float lower = -0.6;
    float upper = 0.1;
    for (int iteration = 0; iteration < 60; ++iteration)
    {
        float middle = 0.5 * (lower + upper);
        float candidate[9] = {};
        candidate[0] = log(density);
        candidate[1] = entropy;
        float fluidVelocity[3] = {};
        for (int axis = 0; axis < 3; ++axis)
            fluidVelocity[axis] = middle * carrierVelocity[axis];
        float fluidSpeedSquared = vtMetricDot3(
            fluidVelocity, geometry.metric, fluidVelocity
        );
        float fluidW = 1.0 / sqrt(
            max(1.0 - fluidSpeedSquared, 1.0e-8)
        );
        for (int axis = 0; axis < 3; ++axis)
            candidate[2 + axis] = fluidW * fluidVelocity[axis];
        candidate[5] = log(carrierDensity);
        for (int axis = 0; axis < 3; ++axis)
            candidate[6 + axis] = carrierW * carrierVelocity[axis];
        VTPrincipal9 principal = vtPrincipal9(
            candidate, geometry, 0, modelMode, -1
        );
        float projection = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            projection +=
                principal.conserved[2 + axis]
                * carrierVelocity[axis];
        }
        if (projection > 0.0)
            upper = middle;
        else
            lower = middle;
    }
    float multiplier = 0.5 * (lower + upper);
    float fluidVelocity[3] = {};
    for (int axis = 0; axis < 3; ++axis)
        fluidVelocity[axis] = multiplier * carrierVelocity[axis];
    float fluidSpeedSquared = vtMetricDot3(
        fluidVelocity, geometry.metric, fluidVelocity
    );
    float fluidW = 1.0 / sqrt(max(1.0 - fluidSpeedSquared, 1.0e-8));
    float finalValues[9] = {};
    finalValues[0] = log(density);
    finalValues[1] = entropy;
    for (int axis = 0; axis < 3; ++axis)
        finalValues[2 + axis] = fluidW * fluidVelocity[axis];
    finalValues[5] = log(carrierDensity);
    for (int axis = 0; axis < 3; ++axis)
        finalValues[6 + axis] = carrierW * carrierVelocity[axis];
    for (int component = 0; component < 9; ++component)
        primitive[point, component] = finalValues[component];
    VTPrincipal9 finalPrincipal = vtPrincipal9(
        finalValues, geometry, 0, modelMode, -1
    );
    float maximumMomentum = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        maximumMomentum = max(
            maximumMomentum,
            abs(
                finalPrincipal.conserved[2 + axis]
                / geometry.sqrtDeterminant
            )
        );
    }
    audit[point, 0] = maximumMomentum;
    audit[point, 1] = finalPrincipal.phaseTwo ? 1.0 : 0.0;
    audit[point, 2] = finalPrincipal.switchingMargin;
}

void theory33CounterflowAuditReduceKernel(
    int point,
    int numberPoints,
    ITensor<float, 2> audit,
    IWTensor<float, 2> control
)
{
    float maximumMomentum = 0.0;
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        maximumMomentum = max(
            maximumMomentum, float(abs(audit[cell, 0]))
        );
    }
    control[0, 11] = maximumMomentum;
}

void theory33CMCStoreSourceKernel(
    int point,
    ITensor<float, 2> matterStress,
    IWTensor<float, 2> storedSource
)
{
    for (int component = 0; component < 4; ++component)
        storedSource[point, component] = matterStress[point, component];
}

void theory33CMCSourceChangeKernel(
    int point,
    int numberPoints,
    ITensor<float, 2> matterStress,
    ITensor<float, 2> storedSource,
    IWTensor<float, 2> control
)
{
    float differenceSquared = 0.0;
    float sourceSquared = 0.0;
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        float current = float(matterStress[cell, 0]);
        float difference =
            current - float(storedSource[cell, 0]);
        differenceSquared += difference * difference;
        sourceSquared += current * current;
    }
    control[0, 10] = sqrt(differenceSquared)
        / max(sqrt(sourceSquared), 1.0e-30);
}

float vtCCZ4InverseSpacing(
    int axis,
    float inverseDx,
    float inverseDy,
    float inverseDz
)
{
    return axis == 0
        ? inverseDx
        : (axis == 1 ? inverseDy : inverseDz);
}

int vtCCZ4AxisCount(int axis, int nx, int ny, int nz)
{
    return axis == 0 ? nx : (axis == 1 ? ny : nz);
}

float vtCCZ4Derivative(
    int point,
    int component,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state
)
{
    if (vtCCZ4AxisCount(axis, nx, ny, nz) <= 1)
        return 0.0;
    float inverse = vtCCZ4InverseSpacing(
        axis, inverseDx, inverseDy, inverseDz
    );
    if (derivativeOrder == 4)
    {
        int minusTwo = vtGridNeighbor(
            point, axis, -2, nx, ny, nz
        );
        int minusOne = vtGridNeighbor(
            point, axis, -1, nx, ny, nz
        );
        int plusOne = vtGridNeighbor(
            point, axis, 1, nx, ny, nz
        );
        int plusTwo = vtGridNeighbor(
            point, axis, 2, nx, ny, nz
        );
        return (inverse / 12.0) * (
            state[minusTwo, component]
            - 8.0 * state[minusOne, component]
            + 8.0 * state[plusOne, component]
            - state[plusTwo, component]
        );
    }
    int plus = vtGridNeighbor(point, axis, 1, nx, ny, nz);
    int minus = vtGridNeighbor(point, axis, -1, nx, ny, nz);
    return 0.5
        * inverse
        * (state[plus, component] - state[minus, component]);
}

float vtCCZ4SecondDerivative(
    int point,
    int component,
    int firstAxis,
    int secondAxis,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state
)
{
    if (
        vtCCZ4AxisCount(firstAxis, nx, ny, nz) <= 1
        || vtCCZ4AxisCount(secondAxis, nx, ny, nz) <= 1
    )
        return 0.0;
    float firstInverse = vtCCZ4InverseSpacing(
        firstAxis, inverseDx, inverseDy, inverseDz
    );
    float secondInverse = vtCCZ4InverseSpacing(
        secondAxis, inverseDx, inverseDy, inverseDz
    );
    if (derivativeOrder == 4)
    {
        int offsets[4] = { -2, -1, 1, 2 };
        float weights[4] = { 1.0, -8.0, 8.0, -1.0 };
        float result = 0.0;
        for (int first = 0; first < 4; ++first)
        {
            int shifted = vtGridNeighbor(
                point, firstAxis, offsets[first], nx, ny, nz
            );
            for (int second = 0; second < 4; ++second)
            {
                int sample = vtGridNeighbor(
                    shifted, secondAxis, offsets[second],
                    nx, ny, nz
                );
                result +=
                    weights[first]
                    * weights[second]
                    * state[sample, component];
            }
        }
        return result
            * firstInverse
            * secondInverse
            / 144.0;
    }
    if (firstAxis == secondAxis)
    {
        int plus = vtGridNeighbor(
            point, firstAxis, 2, nx, ny, nz
        );
        int minus = vtGridNeighbor(
            point, firstAxis, -2, nx, ny, nz
        );
        return 0.25
            * firstInverse
            * firstInverse
            * (
                state[plus, component]
                - 2.0 * state[point, component]
                + state[minus, component]
            );
    }
    int firstPlus = vtGridNeighbor(
        point, firstAxis, 1, nx, ny, nz
    );
    int firstMinus = vtGridNeighbor(
        point, firstAxis, -1, nx, ny, nz
    );
    int pp = vtGridNeighbor(
        firstPlus, secondAxis, 1, nx, ny, nz
    );
    int pm = vtGridNeighbor(
        firstPlus, secondAxis, -1, nx, ny, nz
    );
    int mp = vtGridNeighbor(
        firstMinus, secondAxis, 1, nx, ny, nz
    );
    int mm = vtGridNeighbor(
        firstMinus, secondAxis, -1, nx, ny, nz
    );
    return 0.25
        * firstInverse
        * secondInverse
        * (
            state[pp, component]
            - state[pm, component]
            - state[mp, component]
            + state[mm, component]
        );
}

float vtCCZ4MetricValue(
    int point,
    int row,
    int column,
    bool physical,
    ITensor<float, 2> state
)
{
    float value = state[point, row * 3 + column];
    if (physical)
    {
        float phi = max(state[point, 18], 1.0e-8);
        value /= phi * phi;
    }
    return value;
}

float vtCCZ4MetricDerivative(
    int point,
    int row,
    int column,
    int axis,
    bool physical,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state
)
{
    if (vtCCZ4AxisCount(axis, nx, ny, nz) <= 1)
        return 0.0;
    float inverse = vtCCZ4InverseSpacing(
        axis, inverseDx, inverseDy, inverseDz
    );
    if (derivativeOrder == 4)
    {
        int minusTwo = vtGridNeighbor(
            point, axis, -2, nx, ny, nz
        );
        int minusOne = vtGridNeighbor(
            point, axis, -1, nx, ny, nz
        );
        int plusOne = vtGridNeighbor(
            point, axis, 1, nx, ny, nz
        );
        int plusTwo = vtGridNeighbor(
            point, axis, 2, nx, ny, nz
        );
        return (inverse / 12.0) * (
            vtCCZ4MetricValue(
                minusTwo, row, column, physical, state
            )
            - 8.0 * vtCCZ4MetricValue(
                minusOne, row, column, physical, state
            )
            + 8.0 * vtCCZ4MetricValue(
                plusOne, row, column, physical, state
            )
            - vtCCZ4MetricValue(
                plusTwo, row, column, physical, state
            )
        );
    }
    int plus = vtGridNeighbor(point, axis, 1, nx, ny, nz);
    int minus = vtGridNeighbor(point, axis, -1, nx, ny, nz);
    return 0.5
        * inverse
        * (
            vtCCZ4MetricValue(
                plus, row, column, physical, state
            )
            - vtCCZ4MetricValue(
                minus, row, column, physical, state
            )
        );
}

void vtCCZ4Connection(
    int point,
    bool physical,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state,
    out float metric[9],
    out float inverseMetric[9],
    out float connection[27]
)
{
    float geometryValues[13] = {};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            geometryValues[row * 3 + column] =
                vtCCZ4MetricValue(
                    point, row, column, physical, state
                );
        }
    }
    geometryValues[9] = 1.0;
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    for (int index = 0; index < 9; ++index)
    {
        metric[index] = geometry.metric[index];
        inverseMetric[index] = geometry.inverseMetric[index];
    }
    for (int upper = 0; upper < 3; ++upper)
    {
        for (int first = 0; first < 3; ++first)
        {
            for (int second = 0; second < 3; ++second)
            {
                float value = 0.0;
                for (int inner = 0; inner < 3; ++inner)
                {
                    value += 0.5
                        * inverseMetric[upper * 3 + inner]
                        * (
                            vtCCZ4MetricDerivative(
                                point, inner, second, first,
                                physical, nx, ny, nz,
                                inverseDx, inverseDy, inverseDz,
                                derivativeOrder,
                                state
                            )
                            + vtCCZ4MetricDerivative(
                                point, inner, first, second,
                                physical, nx, ny, nz,
                                inverseDx, inverseDy, inverseDz,
                                derivativeOrder,
                                state
                            )
                            - vtCCZ4MetricDerivative(
                                point, first, second, inner,
                                physical, nx, ny, nz,
                                inverseDx, inverseDy, inverseDz,
                                derivativeOrder,
                                state
                            )
                        );
                }
                connection[upper * 9 + first * 3 + second] = value;
            }
        }
    }
}

float vtCCZ4ConnectionDerivative(
    int point,
    int axis,
    int upper,
    int first,
    int second,
    bool physical,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state
)
{
    if (vtCCZ4AxisCount(axis, nx, ny, nz) <= 1)
        return 0.0;
    int offsets[4] = { -2, -1, 1, 2 };
    float weights[4] = { 1.0, -8.0, 8.0, -1.0 };
    int samples = derivativeOrder == 4 ? 4 : 2;
    float result = 0.0;
    for (int sampleIndex = 0; sampleIndex < samples; ++sampleIndex)
    {
        int offset = derivativeOrder == 4
            ? offsets[sampleIndex]
            : (sampleIndex == 0 ? -1 : 1);
        float weight = derivativeOrder == 4
            ? weights[sampleIndex]
            : (sampleIndex == 0 ? -1.0 : 1.0);
        int sample = vtGridNeighbor(
            point, axis, offset, nx, ny, nz
        );
        float metric[9] = {};
        float inverseMetric[9] = {};
        float connection[27] = {};
        vtCCZ4Connection(
            sample, physical, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state,
            metric, inverseMetric, connection
        );
        result += weight * connection[
            upper * 9 + first * 3 + second
        ];
    }
    return result
        * vtCCZ4InverseSpacing(
            axis, inverseDx, inverseDy, inverseDz
        )
        / (derivativeOrder == 4 ? 12.0 : 2.0);
}

void vtCCZ4ZCovector(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state,
    out float zDown[3]
)
{
    float metric[9] = {};
    float inverseMetric[9] = {};
    float connection[27] = {};
    vtCCZ4Connection(
        point, false, nx, ny, nz,
        inverseDx, inverseDy, inverseDz, derivativeOrder, state,
        metric, inverseMetric, connection
    );
    float contracted[3] = {};
    for (int upper = 0; upper < 3; ++upper)
    {
        for (int first = 0; first < 3; ++first)
        {
            for (int second = 0; second < 3; ++second)
            {
                contracted[upper] +=
                    inverseMetric[first * 3 + second]
                    * connection[upper * 9 + first * 3 + second];
            }
        }
    }
    for (int row = 0; row < 3; ++row)
    {
        zDown[row] = 0.0;
        for (int column = 0; column < 3; ++column)
        {
            zDown[row] += 0.5
                * metric[row * 3 + column]
                * (state[point, 21 + column] - contracted[column]);
        }
    }
}

float vtCCZ4ZDerivative(
    int point,
    int axis,
    int component,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state
)
{
    if (vtCCZ4AxisCount(axis, nx, ny, nz) <= 1)
        return 0.0;
    int offsets[4] = { -2, -1, 1, 2 };
    float weights[4] = { 1.0, -8.0, 8.0, -1.0 };
    int samples = derivativeOrder == 4 ? 4 : 2;
    float result = 0.0;
    for (int sampleIndex = 0; sampleIndex < samples; ++sampleIndex)
    {
        int offset = derivativeOrder == 4
            ? offsets[sampleIndex]
            : (sampleIndex == 0 ? -1 : 1);
        float weight = derivativeOrder == 4
            ? weights[sampleIndex]
            : (sampleIndex == 0 ? -1.0 : 1.0);
        int sample = vtGridNeighbor(
            point, axis, offset, nx, ny, nz
        );
        float zDown[3] = {};
        vtCCZ4ZCovector(
            sample, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state, zDown
        );
        result += weight * zDown[component];
    }
    return result
        * vtCCZ4InverseSpacing(
            axis, inverseDx, inverseDy, inverseDz
        )
        / (derivativeOrder == 4 ? 12.0 : 2.0);
}

void vtCCZ4Ricci(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state,
    out float ricci[9]
)
{
    float metric[9] = {};
    float inverseMetric[9] = {};
    float connection[27] = {};
    vtCCZ4Connection(
        point, true, nx, ny, nz,
        inverseDx, inverseDy, inverseDz, derivativeOrder, state,
        metric, inverseMetric, connection
    );
    for (int first = 0; first < 3; ++first)
    {
        for (int second = 0; second < 3; ++second)
        {
            float value = 0.0;
            for (int upper = 0; upper < 3; ++upper)
            {
                if (vtCCZ4AxisCount(upper, nx, ny, nz) > 1)
                {
                    value += vtCCZ4ConnectionDerivative(
                        point, upper,
                        upper, first, second, true,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    );
                }
                if (vtCCZ4AxisCount(second, nx, ny, nz) > 1)
                {
                    value -= vtCCZ4ConnectionDerivative(
                        point, second,
                        upper, first, upper, true,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    );
                }
                for (int inner = 0; inner < 3; ++inner)
                {
                    value +=
                        connection[upper * 9 + first * 3 + second]
                        * connection[inner * 9 + upper * 3 + inner]
                        - connection[
                            inner * 9 + first * 3 + upper
                        ]
                        * connection[
                            upper * 9 + second * 3 + inner
                        ];
                }
            }
            ricci[first * 3 + second] = value;
        }
    }
    for (int first = 0; first < 3; ++first)
    {
        for (int second = first + 1; second < 3; ++second)
        {
            float symmetric = 0.5
                * (
                    ricci[first * 3 + second]
                    + ricci[second * 3 + first]
                );
            ricci[first * 3 + second] = symmetric;
            ricci[second * 3 + first] = symmetric;
        }
    }
}

void ccz4VacuumRhsKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    float kappa1,
    float kappa2,
    float kappa3,
    float cosmologicalConstant,
    float shiftSpeed,
    float etaShift,
    int slicingMode,
    int evolveShift,
    ITensor<float, 2> state,
    IWTensor<float, 2> rhs
)
{
    float gt[9] = {};
    float gtInverse[9] = {};
    float gtConnection[27] = {};
    vtCCZ4Connection(
        point, false, nx, ny, nz,
        inverseDx, inverseDy, inverseDz, derivativeOrder, state,
        gt, gtInverse, gtConnection
    );
    float h[9] = {};
    float hInverse[9] = {};
    float hConnection[27] = {};
    vtCCZ4Connection(
        point, true, nx, ny, nz,
        inverseDx, inverseDy, inverseDz, derivativeOrder, state,
        h, hInverse, hConnection
    );
    float ricci[9] = {};
    vtCCZ4Ricci(
        point, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        derivativeOrder, state, ricci
    );
    float zDown[3] = {};
    vtCCZ4ZCovector(
        point, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        derivativeOrder, state, zDown
    );
    float phi = max(state[point, 18], 1.0e-8);
    float traceK = state[point, 19];
    float theta = state[point, 20];
    float alpha = state[point, 24];
    float beta[3] = {};
    float betaDivergence = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        beta[axis] = state[point, 25 + axis];
        betaDivergence += vtCCZ4Derivative(
            point, 25 + axis, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state
        );
    }
    float zUpPhysical[3] = {};
    float zUpConformal[3] = {};
    float contractedConformal[3] = {};
    for (int upper = 0; upper < 3; ++upper)
    {
        for (int first = 0; first < 3; ++first)
        {
            zUpPhysical[upper] +=
                hInverse[upper * 3 + first] * zDown[first];
            zUpConformal[upper] +=
                gtInverse[upper * 3 + first] * zDown[first];
            for (int second = 0; second < 3; ++second)
            {
                contractedConformal[upper] +=
                    gtInverse[first * 3 + second]
                    * gtConnection[
                        upper * 9 + first * 3 + second
                    ];
            }
        }
    }
    float dz[9] = {};
    float divZ = 0.0;
    float hessianAlpha[9] = {};
    float laplaceAlpha = 0.0;
    float scalarRicci = 0.0;
    for (int first = 0; first < 3; ++first)
    {
        for (int second = 0; second < 3; ++second)
        {
            float derivativeZ = 0.0;
            if (vtCCZ4AxisCount(first, nx, ny, nz) > 1)
            {
                derivativeZ = vtCCZ4ZDerivative(
                    point, first, second,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
            }
            dz[first * 3 + second] = derivativeZ;
            for (int inner = 0; inner < 3; ++inner)
            {
                dz[first * 3 + second] -=
                    hConnection[inner * 9 + first * 3 + second]
                    * zDown[inner];
            }
            hessianAlpha[first * 3 + second] =
                vtCCZ4SecondDerivative(
                    point, 24, first, second, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
            for (int inner = 0; inner < 3; ++inner)
            {
                hessianAlpha[first * 3 + second] -=
                    hConnection[inner * 9 + first * 3 + second]
                    * vtCCZ4Derivative(
                        point, 24, inner, nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    );
            }
            divZ +=
                hInverse[first * 3 + second]
                * dz[first * 3 + second];
            laplaceAlpha +=
                hInverse[first * 3 + second]
                * hessianAlpha[first * 3 + second];
            scalarRicci +=
                hInverse[first * 3 + second]
                * ricci[first * 3 + second];
        }
    }
    float conformalA[9] = {};
    float aMixed[9] = {};
    float aUp[9] = {};
    float aSquared = 0.0;
    for (int index = 0; index < 9; ++index)
        conformalA[index] = state[point, 9 + index];
    for (int first = 0; first < 3; ++first)
    {
        for (int second = 0; second < 3; ++second)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                aMixed[first * 3 + second] +=
                    gtInverse[first * 3 + inner]
                    * conformalA[inner * 3 + second];
                for (int other = 0; other < 3; ++other)
                {
                    aUp[first * 3 + second] +=
                        gtInverse[first * 3 + inner]
                        * gtInverse[second * 3 + other]
                        * conformalA[inner * 3 + other];
                }
            }
            aSquared +=
                conformalA[first * 3 + second]
                * aUp[first * 3 + second];
        }
    }
    float core[9] = {};
    float coreTrace = 0.0;
    for (int first = 0; first < 3; ++first)
    {
        for (int second = 0; second < 3; ++second)
        {
            core[first * 3 + second] =
                -hessianAlpha[first * 3 + second]
                + alpha
                * (
                    ricci[first * 3 + second]
                    + dz[first * 3 + second]
                    + dz[second * 3 + first]
                );
            coreTrace +=
                hInverse[first * 3 + second]
                * core[first * 3 + second];
        }
    }
    for (int first = 0; first < 3; ++first)
    {
        for (int second = 0; second < 3; ++second)
        {
            float dMetric =
                -2.0 * alpha * conformalA[first * 3 + second]
                - (2.0 / 3.0)
                * gt[first * 3 + second]
                * betaDivergence;
            float dA =
                -(2.0 / 3.0)
                * conformalA[first * 3 + second]
                * betaDivergence;
            for (int axis = 0; axis < 3; ++axis)
            {
                dMetric +=
                    beta[axis]
                    * vtCCZ4Derivative(
                        point, first * 3 + second, axis,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                    + gt[axis * 3 + first]
                    * vtCCZ4Derivative(
                        point, 25 + axis, second,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                    + gt[axis * 3 + second]
                    * vtCCZ4Derivative(
                        point, 25 + axis, first,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    );
                dA +=
                    beta[axis]
                    * vtCCZ4Derivative(
                        point, 9 + first * 3 + second, axis,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                    + conformalA[axis * 3 + first]
                    * vtCCZ4Derivative(
                        point, 25 + axis, second,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                    + conformalA[axis * 3 + second]
                    * vtCCZ4Derivative(
                        point, 25 + axis, first,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    );
            }
            dA += phi
                * phi
                * (
                    core[first * 3 + second]
                    - h[first * 3 + second] * coreTrace / 3.0
                );
            float aa = 0.0;
            for (int inner = 0; inner < 3; ++inner)
            {
                aa +=
                    conformalA[first * 3 + inner]
                    * aMixed[inner * 3 + second];
            }
            dA += alpha
                * (
                    conformalA[first * 3 + second]
                    * (traceK - 2.0 * theta)
                    - 2.0 * aa
                );
            rhs[point, first * 3 + second] = dMetric;
            rhs[point, 9 + first * 3 + second] = dA;
        }
    }
    float advPhi = 0.0;
    float advK = 0.0;
    float advTheta = 0.0;
    float zGradientAlpha = 0.0;
    float advAlpha = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        advPhi += beta[axis] * vtCCZ4Derivative(
            point, 18, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state
        );
        advK += beta[axis] * vtCCZ4Derivative(
            point, 19, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state
        );
        advTheta += beta[axis] * vtCCZ4Derivative(
            point, 20, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state
        );
        advAlpha += beta[axis] * vtCCZ4Derivative(
            point, 24, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state
        );
        zGradientAlpha += zUpPhysical[axis] * vtCCZ4Derivative(
            point, 24, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            derivativeOrder, state
        );
    }
    rhs[point, 18] =
        (alpha * traceK - betaDivergence) * phi / 3.0 + advPhi;
    rhs[point, 19] =
        -laplaceAlpha
        + alpha
        * (
            scalarRicci
            + 2.0 * divZ
            + traceK * traceK
            - 2.0 * theta * traceK
            - 3.0 * kappa1 * (1.0 + kappa2) * theta
            - 3.0 * cosmologicalConstant
        )
        + advK;
    rhs[point, 20] =
        0.5
        * alpha
        * (
            scalarRicci
            + 2.0 * divZ
            - aSquared
            + (2.0 / 3.0) * traceK * traceK
            - 2.0 * theta * traceK
            - 2.0 * cosmologicalConstant
        )
        - zGradientAlpha
        + advTheta
        - alpha * kappa1 * (2.0 + kappa2) * theta;
    float dGammaValues[3] = {};
    for (int upper = 0; upper < 3; ++upper)
    {
        float firstTerm = 0.0;
        for (int first = 0; first < 3; ++first)
        {
            for (int second = 0; second < 3; ++second)
            {
                firstTerm +=
                    gtConnection[upper * 9 + first * 3 + second]
                    * aUp[first * 3 + second];
            }
            firstTerm -=
                3.0
                * aUp[upper * 3 + first]
                * vtCCZ4Derivative(
                    point, 18, first, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                )
                / phi;
            firstTerm -=
                (2.0 / 3.0)
                * gtInverse[upper * 3 + first]
                * vtCCZ4Derivative(
                    point, 19, first, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
        }
        float dGamma = 2.0 * alpha * firstTerm;
        for (int axis = 0; axis < 3; ++axis)
        {
            dGamma +=
                2.0
                * gtInverse[axis * 3 + upper]
                * (
                    alpha
                    * vtCCZ4Derivative(
                        point, 20, axis, nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                    - theta
                    * vtCCZ4Derivative(
                        point, 24, axis, nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                    - (2.0 / 3.0)
                    * alpha
                    * traceK
                    * zDown[axis]
                );
            dGamma -=
                2.0
                * aUp[upper * 3 + axis]
                * vtCCZ4Derivative(
                    point, 24, axis, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
            for (int inner = 0; inner < 3; ++inner)
            {
                dGamma +=
                    gtInverse[axis * 3 + inner]
                    * vtCCZ4SecondDerivative(
                        point, 25 + upper, axis, inner,
                        nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    );
            }
            float gradientDivergence = 0.0;
            for (int inner = 0; inner < 3; ++inner)
            {
                gradientDivergence += vtCCZ4SecondDerivative(
                    point, 25 + inner, axis, inner,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
            }
            dGamma +=
                (1.0 / 3.0)
                * gtInverse[upper * 3 + axis]
                * gradientDivergence;
            dGamma -=
                contractedConformal[axis]
                * vtCCZ4Derivative(
                    point, 25 + upper, axis, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
            dGamma +=
                beta[axis]
                * vtCCZ4Derivative(
                    point, 21 + upper, axis, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
            dGamma +=
                2.0
                * kappa3
                * (
                    (2.0 / 3.0)
                    * zUpConformal[upper]
                    * vtCCZ4Derivative(
                        point, 25 + axis, axis, nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                    - zUpConformal[axis]
                    * vtCCZ4Derivative(
                        point, 25 + upper, axis, nx, ny, nz,
                        inverseDx, inverseDy, inverseDz,
                        derivativeOrder, state
                    )
                );
        }
        dGamma +=
            (2.0 / 3.0)
            * contractedConformal[upper]
            * betaDivergence
            - 2.0 * alpha * kappa1 * zUpConformal[upper];
        dGammaValues[upper] = dGamma;
        rhs[point, 21 + upper] = dGamma;
    }
    rhs[point, 24] =
        (
            slicingMode == 1
            ? -alpha * alpha * (traceK - 2.0 * theta)
            : -2.0 * alpha * (traceK - 2.0 * theta)
        )
        + advAlpha;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (evolveShift != 0)
        {
            float dShift = shiftSpeed * state[point, 28 + axis];
            float dDriver = dGammaValues[axis];
            for (int inner = 0; inner < 3; ++inner)
            {
                dShift += beta[inner] * vtCCZ4Derivative(
                    point, 25 + axis, inner, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
                dDriver -= beta[inner] * vtCCZ4Derivative(
                    point, 21 + axis, inner, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
                dDriver += beta[inner] * vtCCZ4Derivative(
                    point, 28 + axis, inner, nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder, state
                );
            }
            rhs[point, 25 + axis] = dShift;
            rhs[point, 28 + axis] =
                dDriver - etaShift * state[point, 28 + axis];
        }
        else
        {
            rhs[point, 25 + axis] = 0.0;
            rhs[point, 28 + axis] = 0.0;
        }
    }
}

// Compact in-situ visualization output.  This kernel deliberately derives
// observables from the resident CCZ4 state instead of copying the 31-component
// restart ABI.  Columns are:
// alpha, chi, K, Theta, |beta|, Hamiltonian, |Z|, conformal det residual.
void tesseractObserveCCZ4Kernel(
    int point,
    int nx,
    int ny,
    int nz,
    int observationNx,
    int observationNy,
    int observationNz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    float cosmologicalConstant,
    ITensor<float, 2> state,
    IWTensor<float, 2> observation
)
{
    int observationX = point % observationNx;
    int observationY = (point / observationNx) % observationNy;
    int observationZ = point / (observationNx * observationNy);
    int sourceX = min(
        int((float(observationX) + 0.5) * float(nx) / float(observationNx)),
        nx - 1
    );
    int sourceY = min(
        int((float(observationY) + 0.5) * float(ny) / float(observationNy)),
        ny - 1
    );
    int sourceZ = min(
        int((float(observationZ) + 0.5) * float(nz) / float(observationNz)),
        nz - 1
    );
    int sourcePoint = sourceX + nx * (sourceY + ny * sourceZ);
    float gt[9] = {};
    float gtInverse[9] = {};
    float gtConnection[27] = {};
    vtCCZ4Connection(
        sourcePoint, false, nx, ny, nz,
        inverseDx, inverseDy, inverseDz, derivativeOrder, state,
        gt, gtInverse, gtConnection
    );
    float h[9] = {};
    float hInverse[9] = {};
    float hConnection[27] = {};
    vtCCZ4Connection(
        sourcePoint, true, nx, ny, nz,
        inverseDx, inverseDy, inverseDz, derivativeOrder, state,
        h, hInverse, hConnection
    );
    float ricci[9] = {};
    vtCCZ4Ricci(
        sourcePoint, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        derivativeOrder, state, ricci
    );
    float scalarRicci = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
            scalarRicci +=
                hInverse[row * 3 + column] * ricci[row * 3 + column];
    }
    float aSquared = 0.0;
    for (int first = 0; first < 3; ++first)
    {
        for (int second = 0; second < 3; ++second)
        {
            float raised = 0.0;
            for (int inner = 0; inner < 3; ++inner)
            {
                for (int other = 0; other < 3; ++other)
                {
                    raised +=
                        gtInverse[first * 3 + inner]
                        * gtInverse[second * 3 + other]
                        * state[sourcePoint, 9 + inner * 3 + other];
                }
            }
            aSquared += state[sourcePoint, 9 + first * 3 + second] * raised;
        }
    }
    float zDown[3] = {};
    vtCCZ4ZCovector(
        sourcePoint, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        derivativeOrder, state, zDown
    );
    float zSquared = 0.0;
    for (int first = 0; first < 3; ++first)
    {
        for (int second = 0; second < 3; ++second)
            zSquared +=
                hInverse[first * 3 + second]
                * zDown[first]
                * zDown[second];
    }
    float shiftSquared = 0.0;
    for (int axis = 0; axis < 3; ++axis)
        shiftSquared +=
            state[sourcePoint, 25 + axis]
            * state[sourcePoint, 25 + axis];
    float determinant =
        gt[0] * (gt[4] * gt[8] - gt[5] * gt[7])
        - gt[1] * (gt[3] * gt[8] - gt[5] * gt[6])
        + gt[2] * (gt[3] * gt[7] - gt[4] * gt[6]);
    float traceK = state[sourcePoint, 19];
    observation[point, 0] = state[sourcePoint, 24];
    observation[point, 1] = state[sourcePoint, 18];
    observation[point, 2] = traceK;
    observation[point, 3] = state[sourcePoint, 20];
    observation[point, 4] = sqrt(max(shiftSquared, 0.0));
    observation[point, 5] =
        scalarRicci
        - aSquared
        + (2.0 / 3.0) * traceK * traceK
        - 2.0 * cosmologicalConstant;
    observation[point, 6] = sqrt(max(zSquared, 0.0));
    observation[point, 7] = abs(determinant - 1.0);
}

void theory33FullComposeCurvedKernel(
    int point,
    int modelMode,
    ITensor<float, 2> primitive,
    ITensor<float, 1> tracer,
    ITensor<float, 2> geometryInput,
    ITensor<float, 2> vectorStress,
    IWTensor<float, 2> state
)
{
    float x[9] = {};
    float geometryValues[13] = {};
    for (int index = 0; index < 9; ++index)
        x[index] = primitive[point, index];
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    VTPrincipal9 material = vtPrincipal9(
        x, geometry, 0, modelMode, -1
    );
    state[point, 0] = material.conserved[0];
    state[point, 1] = material.conserved[1];
    for (int axis = 0; axis < 3; ++axis)
    {
        state[point, 2 + axis] =
            material.conserved[2 + axis]
            + geometry.sqrtDeterminant
            * vectorStress[point, 1 + axis];
        state[point, 6 + axis] = material.conserved[6 + axis];
    }
    state[point, 5] =
        material.conserved[5]
        + geometry.sqrtDeterminant * vectorStress[point, 0];
    state[point, 9] = material.conserved[0] * x[1];
    state[point, 10] = material.conserved[0] * tracer[point];
}

void theory33CopyVectorStressKernel(
    int point,
    ITensor<float, 2> source,
    IWTensor<float, 2> target
)
{
    for (int component = 0; component < 15; ++component)
        target[point, component] = source[point, component];
}

void theory33FullUpdateVectorStressKernel(
    int point,
    ITensor<float, 2> state,
    ITensor<float, 2> geometryInput,
    ITensor<float, 2> previousVectorStress,
    ITensor<float, 2> currentVectorStress,
    IWTensor<float, 2> output
)
{
    float geometryValues[13] = {};
    for (int component = 0; component < 13; ++component)
        geometryValues[component] = geometryInput[point, component];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    for (int component = 0; component < 11; ++component)
        output[point, component] = state[point, component];
    for (int axis = 0; axis < 3; ++axis)
    {
        output[point, 2 + axis] =
            state[point, 2 + axis]
            + geometry.sqrtDeterminant
            * (
                currentVectorStress[point, 1 + axis]
                - previousVectorStress[point, 1 + axis]
            );
    }
    output[point, 5] =
        state[point, 5]
        + geometry.sqrtDeterminant
        * (
            currentVectorStress[point, 0]
            - previousVectorStress[point, 0]
        );
}

void theory33FullConservativeDragCorrectKernel(
    int point,
    ITensor<float, 2> stateBefore,
    IRWTensor<float, 2> stateAfter
)
{
    // The local implicit drag map changes relative current and thermal
    // entropy, but exactly preserves both densities, total momentum, total
    // energy, and the advected tracer density. Re-impose those invariants
    // after primitive recomposition so float32 recovery error cannot
    // accumulate over long split horizons.
    for (int component = 0; component < 6; ++component)
        stateAfter[point, component] = stateBefore[point, component];
    stateAfter[point, 9] = max(
        stateAfter[point, 9], stateBefore[point, 9]
    );
    stateAfter[point, 10] = stateBefore[point, 10];
}

void theory33FullExtractTracerKernel(
    int point,
    ITensor<float, 2> state,
    IWTensor<float, 1> tracer
)
{
    tracer[point] =
        state[point, 10] / max(state[point, 0], VT_EPSILON);
}

void theory33FullLinearCorridorKernel(
    int point,
    ITensor<float, 2> inputState,
    IWTensor<float, 2> outputState
)
{
    int count = int(inputState.shape[0]);
    float parameter = count > 1
        ? float(point) / float(count - 1)
        : 0.0;
    for (int component = 0; component < 11; ++component)
    {
        outputState[point, component] =
            (1.0 - parameter) * inputState[0, component]
            + parameter * inputState[count - 1, component];
    }
}

void theory33FullRecoverCurvedKernel(
    int point,
    int modelMode,
    float recoveryTolerance,
    float recoveryStep,
    float phaseSwitchTolerance,
    ITensor<float, 2> state,
    ITensor<float, 2> geometryInput,
    ITensor<float, 2> vectorStress,
    ITensor<float, 2> guessInput,
    IWTensor<float, 2> guessOutput,
    IWTensor<float, 2> diagnostics
)
{
    float geometryValues[13] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float target[9] = {};
    target[0] = state[point, 0];
    target[1] = state[point, 1];
    for (int axis = 0; axis < 3; ++axis)
    {
        target[2 + axis] =
            state[point, 2 + axis]
            - geometry.sqrtDeterminant
            * vectorStress[point, 1 + axis];
        target[6 + axis] = state[point, 6 + axis];
    }
    target[5] =
        state[point, 5]
        - geometry.sqrtDeterminant * vectorStress[point, 0];
    float scale[9] = {};
    scale[0] = max(abs(target[0]), 1.0e-8);
    scale[1] = max(abs(target[1]), 1.0e-8);
    float energyScale = max(abs(target[5]), 1.0e-8);
    for (int axis = 0; axis < 3; ++axis)
        energyScale = max(energyScale, abs(target[2 + axis]));
    for (int axis = 0; axis < 3; ++axis)
    {
        scale[2 + axis] = energyScale;
        scale[6 + axis] = max(abs(target[6 + axis]), scale[1]);
    }
    scale[5] = energyScale;
    float working[9] = {};
    for (int index = 0; index < 9; ++index)
        working[index] = guessInput[point, index];
    working[1] = state[point, 9] / max(state[point, 0], VT_EPSILON);
    VTPrincipal9 classified = vtPrincipal9(
        working, geometry, 0, modelMode, -1
    );
    int phaseOverride = modelMode == 1
        ? (classified.phaseTwo ? 1 : 0)
        : -1;
    float finalNorm = 1.0e30;
    int phaseCrossings = 0;
    [MaxIters(3)]
    for (int transition = 0; transition < 3; ++transition)
    {
        [MaxIters(36)]
        for (int iteration = 0; iteration < 36; ++iteration)
        {
            VTPrincipal9 prediction = vtPrincipal9(
                working, geometry, 0, modelMode, phaseOverride
            );
            float residual[9] = {};
            finalNorm = 0.0;
            for (int row = 0; row < 9; ++row)
            {
                residual[row] =
                    (prediction.conserved[row] - target[row]) / scale[row];
                finalNorm = max(finalNorm, abs(residual[row]));
            }
            if (finalNorm < recoveryTolerance)
                break;
            float jacobian[81] = {};
            for (int column = 0; column < 9; ++column)
            {
                float step =
                    recoveryStep * max(1.0, abs(working[column]));
                float plus[9] = {};
                float minus[9] = {};
                for (int index = 0; index < 9; ++index)
                {
                    plus[index] = working[index];
                    minus[index] = working[index];
                }
                plus[column] += step;
                minus[column] -= step;
                VTPrincipal9 plusPrediction = vtPrincipal9(
                    plus, geometry, 0, modelMode, phaseOverride
                );
                VTPrincipal9 minusPrediction = vtPrincipal9(
                    minus, geometry, 0, modelMode, phaseOverride
                );
                for (int row = 0; row < 9; ++row)
                {
                    jacobian[row * 9 + column] =
                        (
                            plusPrediction.conserved[row]
                            - minusPrediction.conserved[row]
                        )
                        / (2.0 * step * scale[row]);
                }
            }
            float rhs[9] = {};
            float delta[9] = {};
            for (int row = 0; row < 9; ++row)
                rhs[row] = -residual[row];
            if (!vtSolve9(jacobian, rhs, delta))
                break;
            bool accepted = false;
            float factor = 1.0;
            [MaxIters(12)]
            for (int lineSearch = 0; lineSearch < 12; ++lineSearch)
            {
                float trial[9] = {};
                for (int index = 0; index < 9; ++index)
                    trial[index] = working[index] + factor * delta[index];
                trial[0] = clamp(trial[0], -20.0, 4.0);
                trial[1] = clamp(trial[1], -20.0, 20.0);
                trial[5] = clamp(trial[5], -20.0, 4.0);
                VTPrincipal9 trialPrediction = vtPrincipal9(
                    trial, geometry, 0, modelMode, phaseOverride
                );
                float trialNorm = 0.0;
                for (int row = 0; row < 9; ++row)
                {
                    trialNorm = max(
                        trialNorm,
                        abs(
                            (
                                trialPrediction.conserved[row]
                                - target[row]
                            )
                            / scale[row]
                        )
                    );
                }
                if (trialNorm < finalNorm)
                {
                    for (int index = 0; index < 9; ++index)
                        working[index] = trial[index];
                    accepted = true;
                    break;
                }
                factor *= 0.5;
            }
            if (!accepted)
                break;
        }
        VTPrincipal9 recovered = vtPrincipal9(
            working, geometry, 0, modelMode, phaseOverride
        );
        // A recovered primitive whose switching margin is inside the
        // float32 nonlinear-solve uncertainty band is located on the
        // event surface. Select the crossed branch just as the float64
        // control does after resolving that event.
        int finalPhase =
            recovered.switchingMargin >= -phaseSwitchTolerance ? 1 : 0;
        if (modelMode != 1 || finalPhase == phaseOverride)
            break;
        phaseOverride = finalPhase;
        phaseCrossings += 1;
    }
    VTPrincipal9 finalPrediction = vtPrincipal9(
        working, geometry, 0, modelMode, phaseOverride
    );
    finalNorm = 0.0;
    for (int row = 0; row < 9; ++row)
    {
        finalNorm = max(
            finalNorm,
            abs(
                (finalPrediction.conserved[row] - target[row]) / scale[row]
            )
        );
    }
    for (int index = 0; index < 9; ++index)
        guessOutput[point, index] = working[index];
    diagnostics[point, 0] = finalNorm;
    diagnostics[point, 1] =
        finalNorm < 5.0e-4 && finalPrediction.valid ? 1.0 : 0.0;
    diagnostics[point, 2] = float(phaseCrossings);
    diagnostics[point, 3] = finalPrediction.phaseTwo ? 1.0 : 0.0;
    diagnostics[point, 4] = finalPrediction.switchingMargin;
    diagnostics[point, 5] =
        state[point, 9] / max(state[point, 0], VT_EPSILON);
    diagnostics[point, 6] =
        state[point, 10] / max(state[point, 0], VT_EPSILON);
    diagnostics[point, 7] = geometry.valid ? 1.0 : 0.0;
}

void theory33FullFluxCurvedKernel(
    int point,
    int axis,
    int modelMode,
    ITensor<float, 2> state,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput,
    ITensor<float, 2> vectorStress,
    IWTensor<float, 2> flux,
    IWTensor<float, 2> pathData,
    IWTensor<float, 1> speed
)
{
    float x[9] = {};
    float geometryValues[13] = {};
    for (int index = 0; index < 9; ++index)
        x[index] = primitive[point, index];
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    VTPrincipal9 material = vtPrincipal9(
        x, geometry, axis, modelMode, -1
    );
    for (int index = 0; index < 9; ++index)
        flux[point, index] = material.flux[index];
    float vectorStressMixed[3] = {};
    for (int component = 0; component < 3; ++component)
    {
        for (int inner = 0; inner < 3; ++inner)
        {
            vectorStressMixed[component] +=
                geometry.inverseMetric[axis * 3 + inner]
                * vectorStress[point, 4 + inner * 3 + component];
        }
        flux[point, 2 + component] =
            material.flux[2 + component]
            + geometry.sqrtDeterminant
            * (
                geometry.lapse * vectorStressMixed[component]
                - geometry.shift[axis]
                * vectorStress[point, 1 + component]
            );
    }
    float vectorMomentumUp = 0.0;
    for (int component = 0; component < 3; ++component)
    {
        vectorMomentumUp +=
            geometry.inverseMetric[axis * 3 + component]
            * vectorStress[point, 1 + component];
    }
    flux[point, 5] =
        material.flux[5]
        + geometry.sqrtDeterminant
        * (
            geometry.lapse * vectorMomentumUp
            - geometry.shift[axis] * vectorStress[point, 0]
        );
    float qN[3] = {};
    qN[0] = x[2];
    qN[1] = x[3];
    qN[2] = x[4];
    float wN = sqrt(
        max(
            1.0 + vtMetricDot3(qN, geometry.metric, qN),
            1.0
        )
    );
    float transportN =
        geometry.lapse * x[2 + axis] / wN - geometry.shift[axis];
    flux[point, 9] = state[point, 9] * transportN;
    flux[point, 10] = state[point, 10] * transportN;
    pathData[point, 0] = material.carrierEulerian;
    pathData[point, 1] = material.chiZero;
    for (int component = 0; component < 3; ++component)
    {
        pathData[point, 2 + component] =
            material.carrierEulerian * material.carrierTransport[component];
        pathData[point, 5 + component] = material.chiDown[component];
    }
    float coordinateLight =
        abs(geometry.shift[axis])
        + geometry.lapse
        * sqrt(max(geometry.inverseMetric[axis * 3 + axis], 0.0));
    speed[point] = 1.35 * coordinateLight;
}

void theory33FullMatterStressKernel(
    int point,
    int modelMode,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput,
    ITensor<float, 2> vectorStress,
    IWTensor<float, 2> matterStress
)
{
    float x[9] = {};
    float geometryValues[13] = {};
    for (int index = 0; index < 9; ++index)
        x[index] = primitive[point, index];
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    float n = exp(clamp(x[0], -20.0, 20.0));
    float d = exp(clamp(x[5], -20.0, 20.0));
    float qN[3] = {};
    float qD[3] = {};
    float velocityN[3] = {};
    float velocityD[3] = {};
    float velocityNDown[3] = {};
    float velocityDDown[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        qN[axis] = x[2 + axis];
        qD[axis] = x[6 + axis];
    }
    float wN = sqrt(
        max(1.0 + vtMetricDot3(qN, geometry.metric, qN), 1.0)
    );
    float wD = sqrt(
        max(1.0 + vtMetricDot3(qD, geometry.metric, qD), 1.0)
    );
    for (int axis = 0; axis < 3; ++axis)
    {
        velocityN[axis] = qN[axis] / wN;
        velocityD[axis] = qD[axis] / wD;
        for (int inner = 0; inner < 3; ++inner)
        {
            velocityNDown[axis] +=
                geometry.metric[axis * 3 + inner] * velocityN[inner];
            velocityDDown[axis] +=
                geometry.metric[axis * 3 + inner] * velocityD[inner];
        }
    }
    float relativeGamma = max(
        wN * wD - vtMetricDot3(qN, geometry.metric, qD), 1.0
    );
    VTConstitutive material = vtConstitutive(
        n,
        d,
        n * d * relativeGamma,
        x[1],
        modelMode,
        -1
    );
    float nW = n * wN;
    float dW = d * wD;
    float momentum[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        momentum[axis] =
            material.bN * nW * nW * velocityNDown[axis]
            + material.bD * dW * dW * velocityDDown[axis]
            + material.entrainment
            * nW
            * dW
            * (velocityNDown[axis] + velocityDDown[axis]);
    }
    matterStress[point, 0] =
        -material.generalizedPressure
        + material.bN * nW * nW
        + material.bD * dW * dW
        + 2.0 * material.entrainment * nW * dW
        + vectorStress[point, 0];
    for (int axis = 0; axis < 3; ++axis)
    {
        matterStress[point, 1 + axis] =
            momentum[axis] + vectorStress[point, 1 + axis];
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            float stress =
                material.generalizedPressure
                * geometry.metric[row * 3 + column]
                + material.bN
                * nW
                * nW
                * velocityNDown[row]
                * velocityNDown[column]
                + material.bD
                * dW
                * dW
                * velocityDDown[row]
                * velocityDDown[column]
                + material.entrainment
                * nW
                * dW
                * (
                    velocityNDown[row] * velocityDDown[column]
                    + velocityDDown[row] * velocityNDown[column]
                );
            matterStress[point, 4 + row * 3 + column] =
                stress + vectorStress[point, 4 + row * 3 + column];
        }
    }
}

void theory33FullInterfaceKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 1> speed,
    IWTensor<float, 2> interfaceFlux
)
{
    int following = vtGridNeighbor(
        point, axis, 1, nx, ny, nz
    );
    float faceSpeed = max(speed[point], speed[following]);
    for (int component = 0; component < 11; ++component)
    {
        interfaceFlux[point, component] =
            0.5 * (flux[point, component] + flux[following, component])
            - 0.5
            * faceSpeed
            * (state[following, component] - state[point, component]);
    }
}

void theory33FullFaceEigensystemKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    int modelMode,
    ITensor<float, 2> primitive,
    ITensor<float, 2> geometryInput,
    IWTensor<float, 2> rightBasis,
    IWTensor<float, 2> leftBasis,
    IWTensor<float, 2> eigenvalues,
    IWTensor<float, 2> pathSymbol,
    IWTensor<float, 2> entropyModes,
    IWTensor<float, 1> transportSpeed,
    IWTensor<float, 2> audit
)
{
    int following = vtGridNeighbor(
        point, axis, 1, nx, ny, nz
    );
    float leftX[9] = {};
    float rightX[9] = {};
    float faceX[9] = {};
    float leftGeometryValues[13] = {};
    float rightGeometryValues[13] = {};
    float faceGeometryValues[13] = {};
    for (int index = 0; index < 9; ++index)
    {
        leftX[index] = primitive[point, index];
        rightX[index] = primitive[following, index];
        faceX[index] = 0.5 * (leftX[index] + rightX[index]);
    }
    for (int index = 0; index < 13; ++index)
    {
        leftGeometryValues[index] = geometryInput[point, index];
        rightGeometryValues[index] = geometryInput[following, index];
        faceGeometryValues[index] = 0.5 * (
            leftGeometryValues[index] + rightGeometryValues[index]
        );
    }
    VTGeometry3 leftGeometry = vtGeometry3(leftGeometryValues);
    VTGeometry3 rightGeometry = vtGeometry3(rightGeometryValues);
    VTGeometry3 faceGeometry = vtGeometry3(faceGeometryValues);
    VTPrincipal9 leftPrediction = vtPrincipal9(
        leftX, leftGeometry, axis, modelMode, -1
    );
    VTPrincipal9 rightPrediction = vtPrincipal9(
        rightX, rightGeometry, axis, modelMode, -1
    );
    int leftPhase = modelMode == 1
        ? (leftPrediction.phaseTwo ? 1 : 0)
        : -1;
    int rightPhase = modelMode == 1
        ? (rightPrediction.phaseTwo ? 1 : 0)
        : -1;
    VTEigensystem9 leftSystem = vtCarterSymbol9(
        leftX, leftGeometry, axis, modelMode, leftPhase
    );
    VTEigensystem9 rightSystem = vtCarterSymbol9(
        rightX, rightGeometry, axis, modelMode, rightPhase
    );
    float averageSymbol[81] = {};
    float selectedRight[81] = {};
    for (int index = 0; index < 81; ++index)
    {
        averageSymbol[index] = 0.5 * (
            leftSystem.symbol[index] + rightSystem.symbol[index]
        );
    }
    VTEigensystem9 averageSystem = vtEigensystem9(averageSymbol);
    float leftPath[81] = {};
    float rightPath[81] = {};
    vtCarterPathSymbol9(
        leftX,
        leftGeometry,
        axis,
        modelMode,
        leftPhase,
        leftPath
    );
    vtCarterPathSymbol9(
        rightX,
        rightGeometry,
        axis,
        modelMode,
        rightPhase,
        rightPath
    );
    for (int index = 0; index < 81; ++index)
    {
        pathSymbol[point, index] =
            0.5 * (leftPath[index] + rightPath[index]);
    }
    bool phaseBoundary =
        modelMode == 1 && leftPhase != rightPhase;
    bool usePhysicalSide =
        phaseBoundary
        || averageSystem.conditionEstimate > 1.0e6
        || !averageSystem.valid;
    bool useLeft =
        leftSystem.conditionEstimate <= rightSystem.conditionEstimate;
    float selectedSymbol[81] = {};
    for (int index = 0; index < 81; ++index)
    {
        selectedSymbol[index] = averageSymbol[index];
        if (usePhysicalSide)
        {
            selectedSymbol[index] = useLeft
                ? leftSystem.symbol[index]
                : rightSystem.symbol[index];
        }
    }
    VTEigensystem9 blockSystem = vtBlockEigensystem9(
        selectedSymbol, axis
    );
    bool selectedValid = usePhysicalSide
        ? (useLeft ? leftSystem.valid : rightSystem.valid)
        : averageSystem.valid;
    float selectedMaximumSpeed = usePhysicalSide
        ? (
            useLeft
            ? leftSystem.maximumSpeed
            : rightSystem.maximumSpeed
        )
        : averageSystem.maximumSpeed;
    bool useBlock =
        blockSystem.valid
        && (
            !selectedValid
            || selectedMaximumSpeed > 1.0 + 2.0e-3
        );

    for (int index = 0; index < 81; ++index)
    {
        float rightValue = averageSystem.right[index];
        float leftValue = averageSystem.left[index];
        if (usePhysicalSide)
        {
            rightValue = useLeft
                ? leftSystem.right[index]
                : rightSystem.right[index];
            leftValue = useLeft
                ? leftSystem.left[index]
                : rightSystem.left[index];
        }
        if (useBlock)
        {
            rightValue = blockSystem.right[index];
            leftValue = blockSystem.left[index];
        }
        selectedRight[index] = rightValue;
        rightBasis[point, index] = rightValue;
        leftBasis[point, index] = leftValue;
    }
    float maximumSpeed = 0.0;
    for (int mode = 0; mode < 9; ++mode)
    {
        float value = averageSystem.values[mode];
        if (usePhysicalSide)
        {
            value = useLeft
                ? leftSystem.values[mode]
                : rightSystem.values[mode];
        }
        if (useBlock)
            value = blockSystem.values[mode];
        eigenvalues[point, mode] = value;
        maximumSpeed = max(maximumSpeed, abs(value));
    }
    float residual = averageSystem.maximumResidual;
    float inverseResidual = averageSystem.inverseResidual;
    float condition = averageSystem.conditionEstimate;
    bool valid = averageSystem.valid;
    if (usePhysicalSide)
    {
        residual = useLeft
            ? leftSystem.maximumResidual
            : rightSystem.maximumResidual;
        inverseResidual = useLeft
            ? leftSystem.inverseResidual
            : rightSystem.inverseResidual;
        condition = useLeft
            ? leftSystem.conditionEstimate
            : rightSystem.conditionEstimate;
        valid = useLeft ? leftSystem.valid : rightSystem.valid;
    }
    if (useBlock)
    {
        residual = blockSystem.maximumResidual;
        inverseResidual = blockSystem.inverseResidual;
        condition = blockSystem.conditionEstimate;
        valid = blockSystem.valid;
    }
    audit[point, 0] = maximumSpeed;
    audit[point, 1] = residual;
    audit[point, 2] = inverseResidual;
    audit[point, 3] = condition;
    audit[point, 4] = valid ? 1.0 : 0.0;
    audit[point, 5] = phaseBoundary ? 1.0 : 0.0;

    float leftTime[81] = {};
    float rightTime[81] = {};
    float leftEntropyJacobian[9] = {};
    float rightEntropyJacobian[9] = {};
    for (int column = 0; column < 9; ++column)
    {
        float leftStep =
            VT_CHARACTERISTIC_STEP9
            * max(1.0, abs(leftX[column]));
        float rightStep =
            VT_CHARACTERISTIC_STEP9
            * max(1.0, abs(rightX[column]));
        float leftPlus[9] = {};
        float leftMinus[9] = {};
        float rightPlus[9] = {};
        float rightMinus[9] = {};
        for (int index = 0; index < 9; ++index)
        {
            leftPlus[index] = leftX[index];
            leftMinus[index] = leftX[index];
            rightPlus[index] = rightX[index];
            rightMinus[index] = rightX[index];
        }
        leftPlus[column] += leftStep;
        leftMinus[column] -= leftStep;
        rightPlus[column] += rightStep;
        rightMinus[column] -= rightStep;
        VTPrincipal9 leftPlusPrediction = vtPrincipal9(
            leftPlus, leftGeometry, axis, modelMode, leftPhase
        );
        VTPrincipal9 leftMinusPrediction = vtPrincipal9(
            leftMinus, leftGeometry, axis, modelMode, leftPhase
        );
        VTPrincipal9 rightPlusPrediction = vtPrincipal9(
            rightPlus, rightGeometry, axis, modelMode, rightPhase
        );
        VTPrincipal9 rightMinusPrediction = vtPrincipal9(
            rightMinus, rightGeometry, axis, modelMode, rightPhase
        );
        float leftInverseStep = 1.0 / (2.0 * leftStep);
        float rightInverseStep = 1.0 / (2.0 * rightStep);
        for (int row = 0; row < 9; ++row)
        {
            leftTime[row * 9 + column] =
                (
                    leftPlusPrediction.conserved[row]
                    - leftMinusPrediction.conserved[row]
                )
                * leftInverseStep;
            rightTime[row * 9 + column] =
                (
                    rightPlusPrediction.conserved[row]
                    - rightMinusPrediction.conserved[row]
                )
                * rightInverseStep;
        }
        leftEntropyJacobian[column] =
            (
                leftPlusPrediction.conserved[0]
                * leftPlus[1]
                - leftMinusPrediction.conserved[0]
                * leftMinus[1]
            )
            * leftInverseStep;
        rightEntropyJacobian[column] =
            (
                rightPlusPrediction.conserved[0]
                * rightPlus[1]
                - rightMinusPrediction.conserved[0]
                * rightMinus[1]
            )
            * rightInverseStep;
    }
    float faceTime[81] = {};
    for (int index = 0; index < 81; ++index)
        faceTime[index] = 0.5 * (leftTime[index] + rightTime[index]);
    float inverseFaceTime[81] = {};
    bool faceTimeValid = vtInvert9(faceTime, inverseFaceTime);
    for (int mode = 0; mode < 9; ++mode)
    {
        float entropyMode = 0.0;
        for (int primitiveRow = 0; primitiveRow < 9; ++primitiveRow)
        {
            float primitiveMode = 0.0;
            for (int physicalRow = 0; physicalRow < 9; ++physicalRow)
            {
                primitiveMode +=
                    inverseFaceTime[
                        primitiveRow * 9 + physicalRow
                    ]
                    * selectedRight[physicalRow * 9 + mode];
            }
            entropyMode +=
                0.5
                * (
                    leftEntropyJacobian[primitiveRow]
                    + rightEntropyJacobian[primitiveRow]
                )
                * primitiveMode;
        }
        entropyModes[point, mode] =
            faceTimeValid ? entropyMode : 0.0;
    }
    float leftQN[3] = {};
    float rightQN[3] = {};
    for (int spatial = 0; spatial < 3; ++spatial)
    {
        leftQN[spatial] = leftX[2 + spatial];
        rightQN[spatial] = rightX[2 + spatial];
    }
    float leftWN = sqrt(
        max(
            1.0 + vtMetricDot3(
                leftQN, leftGeometry.metric, leftQN
            ),
            1.0
        )
    );
    float rightWN = sqrt(
        max(
            1.0 + vtMetricDot3(
                rightQN, rightGeometry.metric, rightQN
            ),
            1.0
        )
    );
    transportSpeed[point] = 0.5 * (
        leftGeometry.lapse * leftQN[axis] / leftWN
        - leftGeometry.shift[axis]
        + rightGeometry.lapse * rightQN[axis] / rightWN
        - rightGeometry.shift[axis]
    );
}

void theory33FullRoeInterfaceKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 1> speed,
    ITensor<float, 2> rightBasis,
    ITensor<float, 2> leftBasis,
    ITensor<float, 2> eigenvalues,
    ITensor<float, 2> entropyModes,
    ITensor<float, 1> transportSpeed,
    IRWTensor<float, 2> interfaceFlux
)
{
    int following = vtGridNeighbor(
        point, axis, 1, nx, ny, nz
    );
    // The full production CPU bound is the coordinate light speed. The
    // robust resident Rusanov baseline stores 1.35 times that value.
    float faceSpeed =
        max(speed[point], speed[following]) / 1.35;
    float characteristicJump[9] = {};
    for (int mode = 0; mode < 9; ++mode)
    {
        for (int component = 0; component < 9; ++component)
        {
            characteristicJump[mode] +=
                leftBasis[point, mode * 9 + component]
                * (
                    state[following, component]
                    - state[point, component]
                );
        }
    }
    for (int component = 0; component < 9; ++component)
    {
        float dissipation = 0.0;
        for (int mode = 0; mode < 9; ++mode)
        {
            float rate =
                0.5 * abs(eigenvalues[point, mode])
                + 0.5 * faceSpeed;
            dissipation +=
                rightBasis[point, component * 9 + mode]
                * rate
                * characteristicJump[mode];
        }
        interfaceFlux[point, component] =
            0.5 * (
                flux[point, component]
                + flux[following, component]
            )
            - 0.5 * dissipation;
    }
    float passiveRate =
        0.5 * abs(transportSpeed[point]) + 0.5 * faceSpeed;
    float entropyPhysicalJump = 0.0;
    float entropyPhysicalDissipation = 0.0;
    float tracerPhysicalJump = 0.0;
    float tracerPhysicalDissipation = 0.0;
    float leftTracer =
        state[point, 10] / max(abs(state[point, 0]), VT_EPSILON);
    float rightTracer =
        state[following, 10]
        / max(abs(state[following, 0]), VT_EPSILON);
    float faceTracer = 0.5 * (leftTracer + rightTracer);
    for (int mode = 0; mode < 9; ++mode)
    {
        float physicalRate =
            0.5 * abs(eigenvalues[point, mode])
            + 0.5 * faceSpeed;
        float entropyMode = entropyModes[point, mode];
        float tracerMode =
            faceTracer * rightBasis[point, mode];
        entropyPhysicalJump +=
            entropyMode * characteristicJump[mode];
        entropyPhysicalDissipation +=
            entropyMode * physicalRate * characteristicJump[mode];
        tracerPhysicalJump +=
            tracerMode * characteristicJump[mode];
        tracerPhysicalDissipation +=
            tracerMode * physicalRate * characteristicJump[mode];
    }
    float entropyPassive =
        state[following, 9]
        - state[point, 9]
        - entropyPhysicalJump;
    float tracerPassive =
        state[following, 10]
        - state[point, 10]
        - tracerPhysicalJump;
    interfaceFlux[point, 9] =
        0.5 * (flux[point, 9] + flux[following, 9])
        - 0.5
        * (
            entropyPhysicalDissipation
            + passiveRate * entropyPassive
        );
    interfaceFlux[point, 10] =
        0.5 * (flux[point, 10] + flux[following, 10])
        - 0.5
        * (
            tracerPhysicalDissipation
            + passiveRate * tracerPassive
        );
}

void theory33FullRoeDirectionalRhsKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseSpacing,
    ITensor<float, 2> state,
    ITensor<float, 2> interfaceFlux,
    ITensor<float, 2> pathSymbol,
    IWTensor<float, 2> rhs
)
{
    int previous = vtGridNeighbor(
        point, axis, -1, nx, ny, nz
    );
    int following = vtGridNeighbor(
        point, axis, 1, nx, ny, nz
    );
    float rightPathJump[9] = {};
    float leftPathJump[9] = {};
    for (int row = 0; row < 9; ++row)
    {
        for (int column = 0; column < 9; ++column)
        {
            rightPathJump[row] +=
                pathSymbol[point, row * 9 + column]
                * (
                    state[following, column]
                    - state[point, column]
                );
            leftPathJump[row] +=
                pathSymbol[previous, row * 9 + column]
                * (
                    state[point, column]
                    - state[previous, column]
                );
        }
    }
    for (int component = 0; component < 11; ++component)
    {
        float pathFluctuation = component < 9
            ? 0.5
            * (
                rightPathJump[component]
                + leftPathJump[component]
            )
            : 0.0;
        rhs[point, component] =
            -(
                interfaceFlux[point, component]
                - interfaceFlux[previous, component]
            )
            * inverseSpacing
            - pathFluctuation * inverseSpacing;
    }
}

float vtFullCarterPathSource(
    int point, int previous, int following, ITensor<float, 2> pathData
)
{
    float densityRight = 0.5 * (pathData[point, 0] + pathData[following, 0]);
    float densityLeft = 0.5 * (pathData[point, 0] + pathData[previous, 0]);
    float source =
        densityRight * (pathData[following, 1] - pathData[point, 1])
        + densityLeft * (pathData[point, 1] - pathData[previous, 1]);
    for (int component = 0; component < 3; ++component)
    {
        int transport = 2 + component;
        int momentum = 5 + component;
        float transportedRight =
            0.5 * (pathData[point, transport] + pathData[following, transport]);
        float transportedLeft =
            0.5 * (pathData[point, transport] + pathData[previous, transport]);
        source += transportedRight * (pathData[following, momentum] - pathData[point, momentum])
            + transportedLeft * (pathData[point, momentum] - pathData[previous, momentum]);
    }
    return source;
}

void theory33FullDirectionalRhsKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseSpacing,
    ITensor<float, 2> interfaceFlux,
    ITensor<float, 2> pathData,
    IWTensor<float, 2> rhs
)
{
    int previous = vtGridNeighbor(
        point, axis, -1, nx, ny, nz
    );
    int following = vtGridNeighbor(
        point, axis, 1, nx, ny, nz
    );
    float pathSource = vtFullCarterPathSource(point, previous, following, pathData);
    for (int component = 0; component < 11; ++component)
    {
        rhs[point, component] =
            -(
                interfaceFlux[point, component]
                - interfaceFlux[previous, component]
            )
            * inverseSpacing
            + (
                component == 6 + axis
                ? 0.5 * inverseSpacing * pathSource
                : 0.0
            );
    }
}

void theory33FullSumRhsKernel(
    int point,
    ITensor<float, 2> rhsX,
    ITensor<float, 2> rhsY,
    ITensor<float, 2> rhsZ,
    IWTensor<float, 2> rhs
)
{
    for (int component = 0; component < 11; ++component)
    {
        rhs[point, component] =
            rhsX[point, component]
            + rhsY[point, component]
            + rhsZ[point, component];
    }
}

void theory33FullZeroRhsKernel(
    int point,
    IWTensor<float, 2> rhs
)
{
    for (int component = 0; component < 11; ++component)
        rhs[point, component] = 0.0;
}

// -------------------------------------------------------------------------
// Conservative block-structured AMR primitives.

int vtAMRLinearIndex(
    int x,
    int y,
    int z,
    int nx,
    int ny
)
{
    return x + nx * (y + ny * z);
}

float vtAMRMinmod(float left, float right)
{
    if (left * right <= 0.0)
        return 0.0;
    return abs(left) < abs(right) ? left : right;
}

void theory33AMRProlongKernel(
    int point,
    int coarseNx,
    int coarseNy,
    int coarseNz,
    int fineNx,
    int fineNy,
    int fineNz,
    int componentCount,
    ITensor<float, 2> coarse,
    IWTensor<float, 2> fine
)
{
    int fineCoordinates[3] = {
        point % fineNx,
        (point / fineNx) % fineNy,
        point / (fineNx * fineNy)
    };
    int coarseCounts[3] = {
        coarseNx, coarseNy, coarseNz
    };
    int parentCoordinates[3] = {
        fineCoordinates[0] / 2,
        fineCoordinates[1] / 2,
        fineCoordinates[2] / 2
    };
    int parent = vtAMRLinearIndex(
        parentCoordinates[0],
        parentCoordinates[1],
        parentCoordinates[2],
        coarseNx,
        coarseNy
    );
    for (int component = 0; component < componentCount; ++component)
    {
        float value = coarse[parent, component];
        for (int axis = 0; axis < 3; ++axis)
        {
            if (coarseCounts[axis] <= 1)
                continue;
            int leftCoordinates[3] = {
                parentCoordinates[0],
                parentCoordinates[1],
                parentCoordinates[2]
            };
            int rightCoordinates[3] = {
                parentCoordinates[0],
                parentCoordinates[1],
                parentCoordinates[2]
            };
            leftCoordinates[axis] = max(
                parentCoordinates[axis] - 1, 0
            );
            rightCoordinates[axis] = min(
                parentCoordinates[axis] + 1,
                coarseCounts[axis] - 1
            );
            int left = vtAMRLinearIndex(
                leftCoordinates[0],
                leftCoordinates[1],
                leftCoordinates[2],
                coarseNx,
                coarseNy
            );
            int right = vtAMRLinearIndex(
                rightCoordinates[0],
                rightCoordinates[1],
                rightCoordinates[2],
                coarseNx,
                coarseNy
            );
            float slope = vtAMRMinmod(
                coarse[parent, component] - coarse[left, component],
                coarse[right, component] - coarse[parent, component]
            );
            float childSign =
                fineCoordinates[axis] % 2 == 0 ? -1.0 : 1.0;
            value += 0.25 * childSign * slope;
        }
        fine[point, component] = value;
    }
}

void theory33AMRRestrictKernel(
    int point,
    int coarseNx,
    int coarseNy,
    int coarseNz,
    int fineNx,
    int fineNy,
    int fineNz,
    int componentCount,
    ITensor<float, 2> fine,
    IWTensor<float, 2> coarse
)
{
    int coarseCoordinates[3] = {
        point % coarseNx,
        (point / coarseNx) % coarseNy,
        point / (coarseNx * coarseNy)
    };
    int activeDimensions =
        (coarseNx > 1 ? 1 : 0)
        + (coarseNy > 1 ? 1 : 0)
        + (coarseNz > 1 ? 1 : 0);
    int children = 1 << activeDimensions;
    for (int component = 0; component < componentCount; ++component)
    {
        float sum = 0.0;
        for (int child = 0; child < children; ++child)
        {
            int bit = 0;
            int fineCoordinates[3] = {};
            int coarseCounts[3] = {
                coarseNx, coarseNy, coarseNz
            };
            for (int axis = 0; axis < 3; ++axis)
            {
                int childOffset = 0;
                if (coarseCounts[axis] > 1)
                {
                    childOffset = (child >> bit) & 1;
                    bit += 1;
                }
                fineCoordinates[axis] =
                    2 * coarseCoordinates[axis] + childOffset;
            }
            int finePoint = vtAMRLinearIndex(
                fineCoordinates[0],
                fineCoordinates[1],
                fineCoordinates[2],
                fineNx,
                fineNy
            );
            sum += fine[finePoint, component];
        }
        coarse[point, component] = sum / float(children);
    }
}

void theory33AMRGaussProjectKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float ZA,
    float ZB,
    ITensor<float, 2> piA,
    ITensor<float, 2> piB,
    IWTensor<float, 2> longitudinal
)
{
    float divergenceA = 0.0;
    float divergenceB = 0.0;
    int counts[3] = { nx, ny, nz };
    float inverseSpacing[3] = {
        inverseDx, inverseDy, inverseDz
    };
    for (int axis = 0; axis < 3; ++axis)
    {
        if (counts[axis] <= 1)
            continue;
        int plus = vtGridNeighbor(
            point, axis, 1, nx, ny, nz
        );
        int minus = vtGridNeighbor(
            point, axis, -1, nx, ny, nz
        );
        divergenceA +=
            0.5
            * inverseSpacing[axis]
            * (piA[plus, axis] - piA[minus, axis])
            / ZA;
        divergenceB +=
            0.5
            * inverseSpacing[axis]
            * (piB[plus, axis] - piB[minus, axis])
            / ZB;
    }
    longitudinal[point, 0] = divergenceA;
    longitudinal[point, 1] = divergenceB;
}

void theory33AMRFluxRegister1DKernel(
    int point,
    int componentCount,
    int lower,
    int upper,
    float dtOverDx,
    ITensor<float, 1> coarseLowerFlux,
    ITensor<float, 1> fineLowerFlux,
    ITensor<float, 1> coarseUpperFlux,
    ITensor<float, 1> fineUpperFlux,
    IWTensor<float, 2> correction
)
{
    for (int component = 0; component < componentCount; ++component)
    {
        float value = 0.0;
        if (point == lower - 1 && lower > 0)
        {
            value -= dtOverDx
                * (
                    fineLowerFlux[component]
                    - coarseLowerFlux[component]
                );
        }
        if (point == upper)
        {
            value += dtOverDx
                * (
                    fineUpperFlux[component]
                    - coarseUpperFlux[component]
                );
        }
        correction[point, component] = value;
    }
}

void theory33AMRRefluxKernel(
    int point,
    int componentCount,
    ITensor<float, 2> coarse,
    ITensor<float, 2> correction,
    IWTensor<float, 2> output
)
{
    for (int component = 0; component < componentCount; ++component)
    {
        output[point, component] =
            coarse[point, component] + correction[point, component];
    }
}

// -------------------------------------------------------------------------
// Nonperiodic physical outer-boundary operators.

#if 1

int vtBoundaryCoordinate(
    int point,
    int axis,
    int nx,
    int ny
)
{
    return axis == 0
        ? point % nx
        : (
            axis == 1
            ? (point / nx) % ny
            : point / (nx * ny)
        );
}

int vtBoundaryCount(int axis, int nx, int ny, int nz)
{
    return axis == 0 ? nx : (axis == 1 ? ny : nz);
}

int vtBoundaryClampedNeighbor(
    int point,
    int axis,
    int offset,
    int nx,
    int ny,
    int nz
)
{
    int x = point % nx;
    int y = (point / nx) % ny;
    int z = point / (nx * ny);
    if (axis == 0)
        x = min(max(x + offset, 0), nx - 1);
    else if (axis == 1)
        y = min(max(y + offset, 0), ny - 1);
    else
        z = min(max(z + offset, 0), nz - 1);
    return x + nx * (y + ny * z);
}

bool vtBoundaryPoint(
    int point,
    int nx,
    int ny,
    int nz
)
{
    int counts[3] = { nx, ny, nz };
    for (int axis = 0; axis < 3; ++axis)
    {
        if (counts[axis] <= 1)
            continue;
        int coordinate = vtBoundaryCoordinate(point, axis, nx, ny);
        if (coordinate == 0 || coordinate == counts[axis] - 1)
            return true;
    }
    return false;
}

float vtBoundaryInverseSpacing(
    int axis,
    float inverseDx,
    float inverseDy,
    float inverseDz
)
{
    return axis == 0
        ? inverseDx
        : (axis == 1 ? inverseDy : inverseDz);
}

float vtBoundaryDerivative(
    int point,
    int component,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> field
)
{
    int count = vtBoundaryCount(axis, nx, ny, nz);
    if (count <= 1)
        return 0.0;
    int coordinate = vtBoundaryCoordinate(point, axis, nx, ny);
    float inverse = vtBoundaryInverseSpacing(
        axis, inverseDx, inverseDy, inverseDz
    );
    if (derivativeOrder == 4 && count >= 5)
    {
        int zero = vtBoundaryClampedNeighbor(
            point, axis, -coordinate, nx, ny, nz
        );
        int one = vtBoundaryClampedNeighbor(
            zero, axis, 1, nx, ny, nz
        );
        int two = vtBoundaryClampedNeighbor(
            zero, axis, 2, nx, ny, nz
        );
        int three = vtBoundaryClampedNeighbor(
            zero, axis, 3, nx, ny, nz
        );
        int four = vtBoundaryClampedNeighbor(
            zero, axis, 4, nx, ny, nz
        );
        if (coordinate == 0)
        {
            return (inverse / 12.0) * (
                -25.0 * field[zero, component]
                + 48.0 * field[one, component]
                - 36.0 * field[two, component]
                + 16.0 * field[three, component]
                - 3.0 * field[four, component]
            );
        }
        if (coordinate == 1)
        {
            return (inverse / 12.0) * (
                -3.0 * field[zero, component]
                - 10.0 * field[one, component]
                + 18.0 * field[two, component]
                - 6.0 * field[three, component]
                + field[four, component]
            );
        }
        if (coordinate == count - 1)
        {
            int last = vtBoundaryClampedNeighbor(
                point, axis, 0, nx, ny, nz
            );
            int backOne = vtBoundaryClampedNeighbor(
                point, axis, -1, nx, ny, nz
            );
            int backTwo = vtBoundaryClampedNeighbor(
                point, axis, -2, nx, ny, nz
            );
            int backThree = vtBoundaryClampedNeighbor(
                point, axis, -3, nx, ny, nz
            );
            int backFour = vtBoundaryClampedNeighbor(
                point, axis, -4, nx, ny, nz
            );
            return -(inverse / 12.0) * (
                -25.0 * field[last, component]
                + 48.0 * field[backOne, component]
                - 36.0 * field[backTwo, component]
                + 16.0 * field[backThree, component]
                - 3.0 * field[backFour, component]
            );
        }
        if (coordinate == count - 2)
        {
            int last = vtBoundaryClampedNeighbor(
                point, axis, 1, nx, ny, nz
            );
            int backOne = point;
            int backTwo = vtBoundaryClampedNeighbor(
                point, axis, -1, nx, ny, nz
            );
            int backThree = vtBoundaryClampedNeighbor(
                point, axis, -2, nx, ny, nz
            );
            int backFour = vtBoundaryClampedNeighbor(
                point, axis, -3, nx, ny, nz
            );
            return -(inverse / 12.0) * (
                -3.0 * field[last, component]
                - 10.0 * field[backOne, component]
                + 18.0 * field[backTwo, component]
                - 6.0 * field[backThree, component]
                + field[backFour, component]
            );
        }
        int plusOne = vtBoundaryClampedNeighbor(
            point, axis, 1, nx, ny, nz
        );
        int plusTwo = vtBoundaryClampedNeighbor(
            point, axis, 2, nx, ny, nz
        );
        int minusOne = vtBoundaryClampedNeighbor(
            point, axis, -1, nx, ny, nz
        );
        int minusTwo = vtBoundaryClampedNeighbor(
            point, axis, -2, nx, ny, nz
        );
        return (inverse / 12.0) * (
            -field[plusTwo, component]
            + 8.0 * field[plusOne, component]
            - 8.0 * field[minusOne, component]
            + field[minusTwo, component]
        );
    }
    if (coordinate == 0 && count >= 3)
    {
        int one = vtBoundaryClampedNeighbor(
            point, axis, 1, nx, ny, nz
        );
        int two = vtBoundaryClampedNeighbor(
            point, axis, 2, nx, ny, nz
        );
        return 0.5 * inverse * (
            -3.0 * field[point, component]
            + 4.0 * field[one, component]
            - field[two, component]
        );
    }
    if (coordinate == count - 1 && count >= 3)
    {
        int one = vtBoundaryClampedNeighbor(
            point, axis, -1, nx, ny, nz
        );
        int two = vtBoundaryClampedNeighbor(
            point, axis, -2, nx, ny, nz
        );
        return 0.5 * inverse * (
            3.0 * field[point, component]
            - 4.0 * field[one, component]
            + field[two, component]
        );
    }
    int plus = vtBoundaryClampedNeighbor(
        point, axis, 1, nx, ny, nz
    );
    int minus = vtBoundaryClampedNeighbor(
        point, axis, -1, nx, ny, nz
    );
    return 0.5 * inverse * (
        field[plus, component] - field[minus, component]
    );
}

void vtBoundaryRadiusNormal(
    int point,
    int nx,
    int ny,
    int nz,
    float dx,
    float dy,
    float dz,
    float originX,
    float originY,
    float originZ,
    out float radius,
    out float normal[3]
)
{
    int coordinates[3] = {
        point % nx,
        (point / nx) % ny,
        point / (nx * ny)
    };
    int counts[3] = { nx, ny, nz };
    float spacing[3] = { dx, dy, dz };
    float origins[3] = { originX, originY, originZ };
    float radiusSquared = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (counts[axis] <= 1)
        {
            normal[axis] = 0.0;
            continue;
        }
        float displacement =
            origins[axis]
            + (float(coordinates[axis]) + 0.5) * spacing[axis];
        normal[axis] = displacement;
        radiusSquared += displacement * displacement;
    }
    radius = sqrt(max(radiusSquared, 1.0e-30));
    for (int axis = 0; axis < 3; ++axis)
        normal[axis] /= radius;
}

float vtBoundaryMetricInverseValue(
    int point,
    int row,
    int column,
    ITensor<float, 2> state
)
{
    float values[13] = {};
    for (int component = 0; component < 9; ++component)
        values[component] = state[point, component];
    values[9] = 1.0;
    VTGeometry3 geometry = vtGeometry3(values);
    return geometry.inverseMetric[3 * row + column];
}

float vtBoundaryMetricInverseDerivative(
    int point,
    int row,
    int column,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state
)
{
    int count = vtBoundaryCount(axis, nx, ny, nz);
    if (count <= 1)
        return 0.0;
    int coordinate = vtBoundaryCoordinate(point, axis, nx, ny);
    float inverse = vtBoundaryInverseSpacing(
        axis, inverseDx, inverseDy, inverseDz
    );
    if (derivativeOrder == 4 && count >= 5)
    {
        int zero = vtBoundaryClampedNeighbor(
            point, axis, -coordinate, nx, ny, nz
        );
        int samples[5] = {
            zero,
            vtBoundaryClampedNeighbor(zero, axis, 1, nx, ny, nz),
            vtBoundaryClampedNeighbor(zero, axis, 2, nx, ny, nz),
            vtBoundaryClampedNeighbor(zero, axis, 3, nx, ny, nz),
            vtBoundaryClampedNeighbor(zero, axis, 4, nx, ny, nz)
        };
        if (coordinate == 0 || coordinate == 1)
        {
            float weights[5] = {};
            if (coordinate == 0)
            {
                weights[0] = -25.0;
                weights[1] = 48.0;
                weights[2] = -36.0;
                weights[3] = 16.0;
                weights[4] = -3.0;
            }
            else
            {
                weights[0] = -3.0;
                weights[1] = -10.0;
                weights[2] = 18.0;
                weights[3] = -6.0;
                weights[4] = 1.0;
            }
            float result = 0.0;
            for (int sample = 0; sample < 5; ++sample)
            {
                result += weights[sample]
                    * vtBoundaryMetricInverseValue(
                        samples[sample], row, column, state
                    );
            }
            return inverse * result / 12.0;
        }
        if (coordinate == count - 1 || coordinate == count - 2)
        {
            int startOffset = coordinate == count - 1 ? 0 : 1;
            int reverseSamples[5] = {
                vtBoundaryClampedNeighbor(
                    point, axis, startOffset, nx, ny, nz
                ),
                vtBoundaryClampedNeighbor(
                    point, axis, startOffset - 1, nx, ny, nz
                ),
                vtBoundaryClampedNeighbor(
                    point, axis, startOffset - 2, nx, ny, nz
                ),
                vtBoundaryClampedNeighbor(
                    point, axis, startOffset - 3, nx, ny, nz
                ),
                vtBoundaryClampedNeighbor(
                    point, axis, startOffset - 4, nx, ny, nz
                )
            };
            float weights[5] = {};
            if (coordinate == count - 1)
            {
                weights[0] = -25.0;
                weights[1] = 48.0;
                weights[2] = -36.0;
                weights[3] = 16.0;
                weights[4] = -3.0;
            }
            else
            {
                weights[0] = -3.0;
                weights[1] = -10.0;
                weights[2] = 18.0;
                weights[3] = -6.0;
                weights[4] = 1.0;
            }
            float result = 0.0;
            for (int sample = 0; sample < 5; ++sample)
            {
                result += weights[sample]
                    * vtBoundaryMetricInverseValue(
                        reverseSamples[sample], row, column, state
                    );
            }
            return -inverse * result / 12.0;
        }
        int plusOne = vtBoundaryClampedNeighbor(
            point, axis, 1, nx, ny, nz
        );
        int plusTwo = vtBoundaryClampedNeighbor(
            point, axis, 2, nx, ny, nz
        );
        int minusOne = vtBoundaryClampedNeighbor(
            point, axis, -1, nx, ny, nz
        );
        int minusTwo = vtBoundaryClampedNeighbor(
            point, axis, -2, nx, ny, nz
        );
        return (inverse / 12.0) * (
            -vtBoundaryMetricInverseValue(
                plusTwo, row, column, state
            )
            + 8.0 * vtBoundaryMetricInverseValue(
                plusOne, row, column, state
            )
            - 8.0 * vtBoundaryMetricInverseValue(
                minusOne, row, column, state
            )
            + vtBoundaryMetricInverseValue(
                minusTwo, row, column, state
            )
        );
    }
    if (coordinate == 0 && count >= 3)
    {
        int one = vtBoundaryClampedNeighbor(
            point, axis, 1, nx, ny, nz
        );
        int two = vtBoundaryClampedNeighbor(
            point, axis, 2, nx, ny, nz
        );
        return 0.5 * inverse * (
            -3.0 * vtBoundaryMetricInverseValue(
                point, row, column, state
            )
            + 4.0 * vtBoundaryMetricInverseValue(
                one, row, column, state
            )
            - vtBoundaryMetricInverseValue(
                two, row, column, state
            )
        );
    }
    if (coordinate == count - 1 && count >= 3)
    {
        int one = vtBoundaryClampedNeighbor(
            point, axis, -1, nx, ny, nz
        );
        int two = vtBoundaryClampedNeighbor(
            point, axis, -2, nx, ny, nz
        );
        return 0.5 * inverse * (
            3.0 * vtBoundaryMetricInverseValue(
                point, row, column, state
            )
            - 4.0 * vtBoundaryMetricInverseValue(
                one, row, column, state
            )
            + vtBoundaryMetricInverseValue(
                two, row, column, state
            )
        );
    }
    int plus = vtBoundaryClampedNeighbor(
        point, axis, 1, nx, ny, nz
    );
    int minus = vtBoundaryClampedNeighbor(
        point, axis, -1, nx, ny, nz
    );
    return 0.5 * inverse * (
        vtBoundaryMetricInverseValue(
            plus, row, column, state
        )
        - vtBoundaryMetricInverseValue(
            minus, row, column, state
        )
    );
}

#endif

#if 1

void theory33CCZ4GammaHatFromMetricKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    ITensor<float, 2> state,
    IWTensor<float, 2> output
)
{
    for (int component = 0; component < 31; ++component)
        output[point, component] = state[point, component];
    for (int upper = 0; upper < 3; ++upper)
    {
        float contracted = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            contracted -= vtBoundaryMetricInverseDerivative(
                point, upper, axis, axis,
                nx, ny, nz,
                inverseDx, inverseDy, inverseDz,
                derivativeOrder,
                state
            );
        }
        output[point, 21 + upper] = contracted;
    }
}

void theory33CCZ4RadiativeBoundaryKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float originX,
    float originY,
    float originZ,
    int derivativeOrder,
    float constraintDamping,
    ITensor<float, 2> state,
    IRWTensor<float, 2> rhs
)
{
    if (!vtBoundaryPoint(point, nx, ny, nz))
        return;
    float radius;
    float normal[3] = {};
    vtBoundaryRadiusNormal(
        point, nx, ny, nz,
        1.0 / inverseDx,
        1.0 / inverseDy,
        1.0 / inverseDz,
        originX, originY, originZ,
        radius, normal
    );
    for (int component = 0; component < 31; ++component)
    {
        float radialDerivative = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            radialDerivative +=
                normal[axis]
                * vtBoundaryDerivative(
                    point, component, axis,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder,
                    state
                );
        }
        float asymptotic =
            (
                component == 0
                || component == 4
                || component == 8
                || component == 18
                || component == 24
            )
            ? 1.0
            : 0.0;
        float speed = component == 24 ? sqrt(2.0) : 1.0;
        rhs[point, component] = -speed * (
            radialDerivative
            + (state[point, component] - asymptotic) / radius
        );
    }

    float values[13] = {};
    for (int component = 0; component < 9; ++component)
        values[component] = state[point, component];
    values[9] = 1.0;
    VTGeometry3 geometry = vtGeometry3(values);
    float contracted[3] = {};
    for (int upper = 0; upper < 3; ++upper)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            contracted[upper] -= vtBoundaryMetricInverseDerivative(
                point, upper, axis, axis,
                nx, ny, nz,
                inverseDx, inverseDy, inverseDz,
                derivativeOrder,
                state
            );
        }
    }
    float zDown[3] = {};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            zDown[row] += 0.5
                * geometry.metric[3 * row + column]
                * (
                    state[point, 21 + column]
                    - contracted[column]
                );
        }
    }
    float zNormal = 0.0;
    float dZNormal = 0.0;
    for (int row = 0; row < 3; ++row)
    {
        zNormal += normal[row] * zDown[row];
        for (int column = 0; column < 3; ++column)
        {
            dZNormal += 0.5
                * normal[row]
                * geometry.metric[3 * row + column]
                * rhs[point, 21 + column];
        }
    }
    float outgoingRhs = rhs[point, 20] - dZNormal;
    float incomingRhs = -constraintDamping
        * (state[point, 20] + zNormal);
    float targetThetaRhs = 0.5 * (outgoingRhs + incomingRhs);
    float targetDZNormal = 0.5 * (incomingRhs - outgoingRhs);
    float delta = targetDZNormal - dZNormal;
    rhs[point, 20] = targetThetaRhs;
    for (int upper = 0; upper < 3; ++upper)
    {
        float correction = 0.0;
        for (int row = 0; row < 3; ++row)
        {
            correction +=
                2.0
                * geometry.inverseMetric[3 * upper + row]
                * normal[row]
                * delta;
        }
        rhs[point, 21 + upper] += correction;
    }
}

#endif

#if 1

void theory33VectorRadiativeBoundaryKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    float originX,
    float originY,
    float originZ,
    int derivativeOrder,
    ITensor<float, 2> state,
    ITensor<float, 2> rawRhs,
    IWTensor<float, 2> projectedRhs
)
{
    for (int component = 0; component < 16; ++component)
        projectedRhs[point, component] = rawRhs[point, component];
    if (!vtBoundaryPoint(point, nx, ny, nz))
        return;
    float radius;
    float normal[3] = {};
    vtBoundaryRadiusNormal(
        point, nx, ny, nz,
        1.0 / inverseDx,
        1.0 / inverseDy,
        1.0 / inverseDz,
        originX, originY, originZ,
        radius, normal
    );
    for (int component = 0; component < 16; ++component)
    {
        if (component == 12 || component == 13)
            continue;
        float radialDerivative = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            radialDerivative +=
                normal[axis]
                * vtBoundaryDerivative(
                    point, component, axis,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    derivativeOrder,
                    state
                );
        }
        projectedRhs[point, component] = -(
            radialDerivative + state[point, component] / radius
        );
    }
}

#endif

#if 1

float vtBoundaryDifferenceDerivative(
    int point,
    int component,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseSpacing,
    int derivativeOrder,
    ITensor<float, 2> rawField,
    ITensor<float, 2> projectedField
)
{
    int count = vtBoundaryCount(axis, nx, ny, nz);
    int coordinate = vtBoundaryCoordinate(point, axis, nx, ny);
    int offsets[5] = {};
    float weights[5] = {};
    int samples = 0;
    float denominator = 2.0;
    if (derivativeOrder == 4 && count >= 5)
    {
        denominator = 12.0;
        samples = 5;
        if (coordinate == 0)
        {
            offsets[0] = 0; offsets[1] = 1; offsets[2] = 2;
            offsets[3] = 3; offsets[4] = 4;
            weights[0] = -25.0; weights[1] = 48.0;
            weights[2] = -36.0; weights[3] = 16.0;
            weights[4] = -3.0;
        }
        else if (coordinate == 1)
        {
            offsets[0] = -1; offsets[1] = 0; offsets[2] = 1;
            offsets[3] = 2; offsets[4] = 3;
            weights[0] = -3.0; weights[1] = -10.0;
            weights[2] = 18.0; weights[3] = -6.0;
            weights[4] = 1.0;
        }
        else if (coordinate == count - 1)
        {
            offsets[0] = 0; offsets[1] = -1; offsets[2] = -2;
            offsets[3] = -3; offsets[4] = -4;
            weights[0] = 25.0; weights[1] = -48.0;
            weights[2] = 36.0; weights[3] = -16.0;
            weights[4] = 3.0;
        }
        else if (coordinate == count - 2)
        {
            offsets[0] = 1; offsets[1] = 0; offsets[2] = -1;
            offsets[3] = -2; offsets[4] = -3;
            weights[0] = 3.0; weights[1] = 10.0;
            weights[2] = -18.0; weights[3] = 6.0;
            weights[4] = -1.0;
        }
        else
        {
            offsets[0] = -2; offsets[1] = -1; offsets[2] = 0;
            offsets[3] = 1; offsets[4] = 2;
            weights[0] = 1.0; weights[1] = -8.0;
            weights[2] = 0.0; weights[3] = 8.0;
            weights[4] = -1.0;
        }
    }
    else
    {
        samples = 3;
        if (coordinate == 0)
        {
            offsets[0] = 0; offsets[1] = 1; offsets[2] = 2;
            weights[0] = -3.0; weights[1] = 4.0;
            weights[2] = -1.0;
        }
        else if (coordinate == count - 1)
        {
            offsets[0] = 0; offsets[1] = -1; offsets[2] = -2;
            weights[0] = 3.0; weights[1] = -4.0;
            weights[2] = 1.0;
        }
        else
        {
            offsets[0] = -1; offsets[1] = 0; offsets[2] = 1;
            weights[0] = -1.0; weights[1] = 0.0;
            weights[2] = 1.0;
        }
    }
    float result = 0.0;
    for (int sample = 0; sample < samples; ++sample)
    {
        int neighbor = vtBoundaryClampedNeighbor(
            point, axis, offsets[sample], nx, ny, nz
        );
        result += weights[sample]
            * (
                projectedField[neighbor, component]
                - rawField[neighbor, component]
            );
    }
    return inverseSpacing * result / denominator;
}

void theory33VectorGaussBoundaryKernel(
    int point,
    int nx,
    int ny,
    int nz,
    float inverseDx,
    float inverseDy,
    float inverseDz,
    int derivativeOrder,
    float ZA,
    float ZB,
    ITensor<float, 2> rawRhs,
    ITensor<float, 2> projectedRhs,
    IWTensor<float, 2> finalRhs
)
{
    for (int component = 0; component < 16; ++component)
        finalRhs[point, component] = projectedRhs[point, component];
    float correctionA = 0.0;
    float correctionB = 0.0;
    int counts[3] = { nx, ny, nz };
    float inverseSpacing[3] = {
        inverseDx, inverseDy, inverseDz
    };
    for (int axis = 0; axis < 3; ++axis)
    {
        if (counts[axis] <= 1)
            continue;
        int componentA = 3 + axis;
        int componentB = 9 + axis;
        float derivativeA = vtBoundaryDifferenceDerivative(
            point, componentA, axis, nx, ny, nz,
            inverseSpacing[axis], derivativeOrder,
            rawRhs, projectedRhs
        );
        float derivativeB = vtBoundaryDifferenceDerivative(
            point, componentB, axis, nx, ny, nz,
            inverseSpacing[axis], derivativeOrder,
            rawRhs, projectedRhs
        );
        correctionA += derivativeA / ZA;
        correctionB += derivativeB / ZB;
    }
    finalRhs[point, 12] = rawRhs[point, 12] + correctionA;
    finalRhs[point, 13] = rawRhs[point, 13] + correctionB;
}

#endif

#if 1

void theory33LinearBoundaryFlux2Kernel(
    int point,
    int outwardSign,
    ITensor<float, 2> matrix,
    ITensor<float, 2> state,
    IWTensor<float, 2> flux
)
{
    float a = matrix[point, 0];
    float b = matrix[point, 1];
    float c = matrix[point, 2];
    float d = matrix[point, 3];
    float discriminant = max(
        (a - d) * (a - d) + 4.0 * b * c, 0.0
    );
    float root = sqrt(discriminant);
    float eigenvalues[2] = {};
    eigenvalues[0] = 0.5 * (a + d - root);
    eigenvalues[1] = 0.5 * (a + d + root);
    float right[4] = {};
    for (int mode = 0; mode < 2; ++mode)
    {
        float lambda = eigenvalues[mode];
        if (abs(b) >= abs(c) && abs(b) > 1.0e-12)
        {
            right[mode] = b;
            right[2 + mode] = lambda - a;
        }
        else if (abs(c) > 1.0e-12)
        {
            right[mode] = lambda - d;
            right[2 + mode] = c;
        }
        else
        {
            right[mode] = mode == 0 ? 1.0 : 0.0;
            right[2 + mode] = mode == 0 ? 0.0 : 1.0;
        }
    }
    float determinant =
        right[0] * right[3] - right[1] * right[2];
    if (abs(determinant) < 1.0e-12)
    {
        flux[point, 0] =
            outwardSign * a >= 0.0 ? a * state[point, 0] : 0.0;
        flux[point, 1] =
            outwardSign * d >= 0.0 ? d * state[point, 1] : 0.0;
        return;
    }
    float inverse[4] = {};
    inverse[0] = right[3] / determinant;
    inverse[1] = -right[1] / determinant;
    inverse[2] = -right[2] / determinant;
    inverse[3] = right[0] / determinant;
    float modes[2] = {};
    modes[0] =
        inverse[0] * state[point, 0]
        + inverse[1] * state[point, 1];
    modes[1] =
        inverse[2] * state[point, 0]
        + inverse[3] * state[point, 1];
    for (int mode = 0; mode < 2; ++mode)
    {
        if (outwardSign * eigenvalues[mode] < 0.0)
            modes[mode] = 0.0;
        else
            modes[mode] *= eigenvalues[mode];
    }
    flux[point, 0] =
        right[0] * modes[0] + right[1] * modes[1];
    flux[point, 1] =
        right[2] * modes[0] + right[3] * modes[1];
}

#endif

void theory33FullOutflowBoundaryFluxKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    ITensor<float, 2> state,
    ITensor<float, 2> primitive,
    ITensor<float, 2> flux,
    IWTensor<float, 2> boundaryFlux
)
{
    int coordinate = vtBoundaryCoordinate(point, axis, nx, ny);
    int count = vtBoundaryCount(axis, nx, ny, nz);
    bool lower = coordinate == 0;
    bool upper = coordinate == count - 1;
    float transport = state[point, 0] != 0.0
        ? flux[point, 0] / state[point, 0]
        : primitive[point, 2 + axis];
    bool incoming =
        (lower && transport > 0.0)
        || (upper && transport < 0.0);
    for (int component = 0; component < 11; ++component)
    {
        boundaryFlux[point, component] =
            (lower || upper) && !incoming
            ? flux[point, component]
            : 0.0;
    }
}

void theory33FullOutflowDirectionalRhsKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseSpacing,
    ITensor<float, 2> interfaceFlux,
    ITensor<float, 2> boundaryFlux,
    ITensor<float, 2> pathData,
    IWTensor<float, 2> rhs
)
{
    int coordinate = vtBoundaryCoordinate(point, axis, nx, ny);
    int count = vtBoundaryCount(axis, nx, ny, nz);
    int previous = vtBoundaryClampedNeighbor(
        point, axis, -1, nx, ny, nz
    );
    int following = vtBoundaryClampedNeighbor(
        point, axis, 1, nx, ny, nz
    );
    float pathSource = vtFullCarterPathSource(point, previous, following, pathData);
    for (int component = 0; component < 11; ++component)
    {
        float rightFlux = coordinate == count - 1
            ? boundaryFlux[point, component]
            : interfaceFlux[point, component];
        float leftFlux = coordinate == 0
            ? boundaryFlux[point, component]
            : interfaceFlux[previous, component];
        rhs[point, component] =
            -(rightFlux - leftFlux) * inverseSpacing
            + (
                component == 6 + axis
                ? 0.5 * inverseSpacing * pathSource
                : 0.0
            );
    }
}

void theory33FullRoeOutflowDirectionalRhsKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    float inverseSpacing,
    ITensor<float, 2> state,
    ITensor<float, 2> interfaceFlux,
    ITensor<float, 2> boundaryFlux,
    ITensor<float, 2> pathSymbol,
    IWTensor<float, 2> rhs
)
{
    int coordinate = vtBoundaryCoordinate(point, axis, nx, ny);
    int count = vtBoundaryCount(axis, nx, ny, nz);
    int previous = vtBoundaryClampedNeighbor(
        point, axis, -1, nx, ny, nz
    );
    int following = vtBoundaryClampedNeighbor(
        point, axis, 1, nx, ny, nz
    );
    float rightPathJump[9] = {};
    float leftPathJump[9] = {};
    for (int row = 0; row < 9; ++row)
    {
        for (int column = 0; column < 9; ++column)
        {
            rightPathJump[row] +=
                pathSymbol[point, row * 9 + column]
                * (
                    state[following, column]
                    - state[point, column]
                );
            leftPathJump[row] +=
                pathSymbol[previous, row * 9 + column]
                * (
                    state[point, column]
                    - state[previous, column]
                );
        }
    }
    for (int component = 0; component < 11; ++component)
    {
        float rightFlux = coordinate == count - 1
            ? boundaryFlux[point, component]
            : interfaceFlux[point, component];
        float leftFlux = coordinate == 0
            ? boundaryFlux[point, component]
            : interfaceFlux[previous, component];
        float pathFluctuation = component < 9
            ? 0.5
            * (
                rightPathJump[component]
                + leftPathJump[component]
            )
            : 0.0;
        rhs[point, component] =
            -(rightFlux - leftFlux) * inverseSpacing
            - pathFluctuation * inverseSpacing;
    }
}

void theory33FullStageKernel(
    int point,
    float dt,
    float baseWeight,
    float evolvedWeight,
    ITensor<float, 2> baseState,
    ITensor<float, 2> stageState,
    ITensor<float, 2> rhs,
    IWTensor<float, 2> output
)
{
    for (int component = 0; component < 11; ++component)
    {
        float evolved =
            stageState[point, component] + dt * rhs[point, component];
        output[point, component] =
            baseWeight * baseState[point, component]
            + evolvedWeight * evolved;
    }
}

void theory33FullProjectEntropyKernel(
    int point,
    ITensor<float, 2> primitive,
    ITensor<float, 2> stateInput,
    IWTensor<float, 2> stateOutput
)
{
    for (int component = 0; component < 11; ++component)
        stateOutput[point, component] = stateInput[point, component];
    // Entropy is already an evolved conservative density. Recovery uses its
    // specific value as the thermodynamic coordinate; overwriting it with a
    // float32 recovered product at every RK stage creates secular entropy
    // loss. Physical entropy production is applied by the implicit drag map.
}

float vtRelativeGammaCurved(
    in float x[9],
    in VTGeometry3 geometry
)
{
    float qN[3] = {};
    float qD[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        qN[axis] = x[2 + axis];
        qD[axis] = x[6 + axis];
    }
    float wN = sqrt(
        max(1.0 + vtMetricDot3(qN, geometry.metric, qN), 1.0)
    );
    float wD = sqrt(
        max(1.0 + vtMetricDot3(qD, geometry.metric, qD), 1.0)
    );
    return max(
        wN * wD - vtMetricDot3(qN, geometry.metric, qD), 1.0
    );
}

float vtDragCurvedResidual(
    VTPrincipal9 prediction,
    in float x[9],
    in float target[9],
    in float scale[9],
    out float residual[9]
)
{
    float norm = 0.0;
    for (int row = 0; row < 6; ++row)
    {
        residual[row] =
            (prediction.conserved[row] - target[row]) / scale[row];
        norm = max(norm, abs(residual[row]));
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        residual[6 + axis] =
            (x[6 + axis] - x[2 + axis] - target[6 + axis])
            / scale[6 + axis];
        norm = max(norm, abs(residual[6 + axis]));
    }
    return norm;
}

void theory33FullDragCurvedKernel(
    int point,
    int modelMode,
    float dt,
    float relaxationTime,
    ITensor<float, 2> geometryInput,
    ITensor<float, 2> primitiveInput,
    IWTensor<float, 2> primitiveOutput,
    IWTensor<float, 2> diagnostics
)
{
    float geometryValues[13] = {};
    float initial[9] = {};
    for (int index = 0; index < 13; ++index)
        geometryValues[index] = geometryInput[point, index];
    for (int index = 0; index < 9; ++index)
        initial[index] = primitiveInput[point, index];
    VTGeometry3 geometry = vtGeometry3(geometryValues);
    VTPrincipal9 before = vtPrincipal9(
        initial, geometry, 0, modelMode, -1
    );
    float oldGamma = vtRelativeGammaCurved(initial, geometry);
    if (oldGamma - 1.0 <= 1.0e-3)
    {
        for (int index = 0; index < 9; ++index)
            primitiveOutput[point, index] = initial[index];
        diagnostics[point, 0] = oldGamma;
        diagnostics[point, 1] = oldGamma;
        diagnostics[point, 2] = 0.0;
        diagnostics[point, 3] = 0.0;
        diagnostics[point, 4] = 1.0;
        diagnostics[point, 5] = 0.0;
        return;
    }
    float decay = exp(
        -dt
        * max(geometry.lapse, 0.0)
        / max(relaxationTime, VT_EPSILON)
    );
    float target[9] = {};
    float scale[9] = {};
    for (int row = 0; row < 6; ++row)
        target[row] = before.conserved[row];
    for (int axis = 0; axis < 3; ++axis)
    {
        target[6 + axis] =
            decay * (initial[6 + axis] - initial[2 + axis]);
    }
    float energyScale = max(abs(before.conserved[5]), 1.0e-8);
    for (int axis = 0; axis < 3; ++axis)
        energyScale = max(energyScale, abs(before.conserved[2 + axis]));
    scale[0] = max(abs(before.conserved[0]), 1.0e-8);
    scale[1] = max(abs(before.conserved[1]), 1.0e-8);
    for (int row = 2; row < 6; ++row)
        scale[row] = energyScale;
    scale[6] = 1.0;
    scale[7] = 1.0;
    scale[8] = 1.0;
    float working[9] = {};
    for (int index = 0; index < 9; ++index)
        working[index] = initial[index];
    int phaseOverride = modelMode == 1
        ? (before.phaseTwo ? 1 : 0)
        : -1;
    float finalNorm = 1.0e30;
    int phaseCrossings = 0;
    [MaxIters(3)]
    for (int transition = 0; transition < 3; ++transition)
    {
        [MaxIters(28)]
        for (int iteration = 0; iteration < 28; ++iteration)
        {
            VTPrincipal9 prediction = vtPrincipal9(
                working, geometry, 0, modelMode, phaseOverride
            );
            float residual[9] = {};
            finalNorm = vtDragCurvedResidual(
                prediction, working, target, scale, residual
            );
            if (finalNorm < 3.0e-5)
                break;
            float jacobian[81] = {};
            for (int column = 0; column < 9; ++column)
            {
                float step =
                    1.5e-3 * max(1.0, abs(working[column]));
                float plus[9] = {};
                float minus[9] = {};
                for (int index = 0; index < 9; ++index)
                {
                    plus[index] = working[index];
                    minus[index] = working[index];
                }
                plus[column] += step;
                minus[column] -= step;
                VTPrincipal9 plusPrediction = vtPrincipal9(
                    plus, geometry, 0, modelMode, phaseOverride
                );
                VTPrincipal9 minusPrediction = vtPrincipal9(
                    minus, geometry, 0, modelMode, phaseOverride
                );
                float plusResidual[9] = {};
                float minusResidual[9] = {};
                vtDragCurvedResidual(
                    plusPrediction,
                    plus,
                    target,
                    scale,
                    plusResidual
                );
                vtDragCurvedResidual(
                    minusPrediction,
                    minus,
                    target,
                    scale,
                    minusResidual
                );
                for (int row = 0; row < 9; ++row)
                {
                    jacobian[row * 9 + column] =
                        (plusResidual[row] - minusResidual[row])
                        / (2.0 * step);
                }
            }
            float rhs[9] = {};
            float delta[9] = {};
            for (int row = 0; row < 9; ++row)
                rhs[row] = -residual[row];
            if (!vtSolve9(jacobian, rhs, delta))
                break;
            bool accepted = false;
            float factor = 1.0;
            [MaxIters(12)]
            for (int lineSearch = 0; lineSearch < 12; ++lineSearch)
            {
                float trial[9] = {};
                for (int index = 0; index < 9; ++index)
                    trial[index] = working[index] + factor * delta[index];
                trial[0] = clamp(trial[0], -20.0, 4.0);
                trial[1] = clamp(trial[1], -20.0, 20.0);
                trial[5] = clamp(trial[5], -20.0, 4.0);
                VTPrincipal9 trialPrediction = vtPrincipal9(
                    trial, geometry, 0, modelMode, phaseOverride
                );
                float trialResidual[9] = {};
                float trialNorm = vtDragCurvedResidual(
                    trialPrediction,
                    trial,
                    target,
                    scale,
                    trialResidual
                );
                if (trialNorm < finalNorm)
                {
                    for (int index = 0; index < 9; ++index)
                        working[index] = trial[index];
                    accepted = true;
                    break;
                }
                factor *= 0.5;
            }
            if (!accepted)
                break;
        }
        VTPrincipal9 recovered = vtPrincipal9(
            working, geometry, 0, modelMode, phaseOverride
        );
        int finalPhase = recovered.switchingMargin >= 0.0 ? 1 : 0;
        if (modelMode != 1 || finalPhase == phaseOverride)
            break;
        phaseOverride = finalPhase;
        phaseCrossings += 1;
    }
    VTPrincipal9 after = vtPrincipal9(
        working, geometry, 0, modelMode, phaseOverride
    );
    float finalResidual[9] = {};
    finalNorm = vtDragCurvedResidual(
        after, working, target, scale, finalResidual
    );
    float newGamma = vtRelativeGammaCurved(working, geometry);
    float entropyIncrease = working[1] - initial[1];
    bool valid =
        finalNorm < 5.0e-4
        && entropyIncrease >= -2.0e-5
        && newGamma <= oldGamma + 1.0e-5;
    for (int index = 0; index < 9; ++index)
        primitiveOutput[point, index] = working[index];
    diagnostics[point, 0] = oldGamma;
    diagnostics[point, 1] = newGamma;
    diagnostics[point, 2] = entropyIncrease;
    diagnostics[point, 3] = finalNorm;
    diagnostics[point, 4] = valid ? 1.0 : 0.0;
    diagnostics[point, 5] = float(phaseCrossings);
}

void vtWriteDiagnostic(
    int point,
    VTRecovery recovery,
    IWTensor<float, 2> diagnostics
)
{
    diagnostics[point, 0] = exp(recovery.x[0]);
    diagnostics[point, 1] = recovery.prediction.velocityN;
    diagnostics[point, 2] = exp(recovery.x[2]);
    diagnostics[point, 3] = recovery.prediction.velocityD;
    diagnostics[point, 4] = recovery.x[4];
    diagnostics[point, 5] = recovery.residual;
    diagnostics[point, 6] = recovery.valid ? 1.0 : 0.0;
    diagnostics[point, 7] = recovery.prediction.relativeGamma;
    diagnostics[point, 8] = recovery.prediction.legendreMinimum;
    diagnostics[point, 9] = recovery.prediction.thermodynamicMinimum;
    diagnostics[point, 10] = recovery.prediction.phaseTwo ? 1.0 : 0.0;
    diagnostics[point, 11] = recovery.prediction.phaseThreshold;
    diagnostics[point, 12] = recovery.prediction.switchingMargin;
    diagnostics[point, 13] = float(recovery.phaseCrossings);
}

void initializeTheory33Kernel(
    int point,
    int modelMode,
    ITensor<float, 2> primitiveInput,
    IWTensor<float, 2> state,
    IWTensor<float, 2> recoveryGuess,
    IWTensor<float, 2> diagnostics
)
{
    float n = max(primitiveInput[point, 0], VT_DENSITY_FLOOR);
    float internal = max(primitiveInput[point, 1], 1.0e-8);
    float velocityN = clamp(primitiveInput[point, 2], -0.95, 0.95);
    float d = max(primitiveInput[point, 3], VT_CARRIER_FLOOR);
    float velocityD = clamp(primitiveInput[point, 4], -0.95, 0.95);
    float tracer = primitiveInput[point, 5];
    float pressure = (VT_GAMMA_AD - 1.0) * n * internal;
    float entropy =
        log(max(pressure / pow(n, VT_GAMMA_AD), VT_EPSILON))
        / (VT_GAMMA_AD - 1.0);
    float x[VT_RECOVERY_WIDTH] = {};
    x[0] = log(n);
    x[1] = velocityN / sqrt(max(1.0 - velocityN * velocityN, VT_EPSILON));
    x[2] = log(d);
    x[3] = velocityD / sqrt(max(1.0 - velocityD * velocityD, VT_EPSILON));
    x[4] = entropy;
    VTPrediction prediction = vtPredict(x, modelMode, -1);
    for (int column = 0; column < 5; ++column)
        state[point, column] = prediction.conserved[column];
    state[point, 5] = prediction.conserved[0] * entropy;
    state[point, 6] = prediction.conserved[0] * tracer;
    for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
        recoveryGuess[point, column] = x[column];

    VTRecovery recovery = {};
    for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
        recovery.x[column] = x[column];
    recovery.prediction = prediction;
    recovery.residual = 0.0;
    recovery.valid = true;
    recovery.phaseCrossings = 0;
    vtWriteDiagnostic(point, recovery, diagnostics);
}

void recoverTheory33Kernel(
    int point,
    int modelMode,
    ITensor<float, 2> state,
    ITensor<float, 2> guessInput,
    IWTensor<float, 2> guessOutput,
    IWTensor<float, 2> diagnostics
)
{
    float target[VT_RECOVERY_WIDTH] = {};
    float initial[VT_RECOVERY_WIDTH] = {};
    for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
    {
        target[column] = state[point, column];
        initial[column] = guessInput[point, column];
    }
    VTRecovery recovery = vtRecover(target, initial, modelMode);
    for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
        guessOutput[point, column] = recovery.x[column];
    vtWriteDiagnostic(point, recovery, diagnostics);
}

void projectTheory33Kernel(
    int point,
    ITensor<float, 2> recoveryGuess,
    IRWTensor<float, 2> state
)
{
    state[point, 5] =
        state[point, 0] * recoveryGuess[point, 4];
}

void theory33FluxKernel(
    int point,
    int modelMode,
    ITensor<float, 2> state,
    ITensor<float, 2> recoveryGuess,
    IWTensor<float, 2> flux,
    IWTensor<float, 2> pathData
)
{
    float x[VT_RECOVERY_WIDTH] = {};
    for (int column = 0; column < VT_RECOVERY_WIDTH; ++column)
        x[column] = recoveryGuess[point, column];
    VTPrediction prediction = vtPredict(x, modelMode, -1);
    for (int column = 0; column < 5; ++column)
        flux[point, column] = prediction.flux[column];
    flux[point, 5] =
        prediction.conserved[0]
        * prediction.entropy
        * prediction.velocityN;
    flux[point, 6] = state[point, 6] * prediction.velocityN;
    pathData[point, 0] = prediction.carrierEulerian;
    pathData[point, 1] = prediction.chiZero;
    pathData[point, 2] = prediction.carrierTransported;
    pathData[point, 3] = prediction.chiDown;
}

void theory33EigensystemKernel(
    int point,
    int modelMode,
    ITensor<float, 2> recoveryGuess,
    IWTensor<float, 2> rightBasis,
    IWTensor<float, 2> leftBasis,
    IWTensor<float, 2> eigenvalues,
    IWTensor<float, 2> audit
)
{
    int count = int(recoveryGuess.shape[0]);
    int following = point + 1 == count ? 0 : point + 1;
    float leftX[5] = {};
    float rightX[5] = {};
    float faceX[5] = {};
    for (int column = 0; column < 5; ++column)
    {
        leftX[column] = recoveryGuess[point, column];
        rightX[column] = recoveryGuess[following, column];
        faceX[column] = 0.5 * (
            leftX[column] + rightX[column]
        );
    }
    VTPrediction leftPrediction = vtPredict(leftX, modelMode, -1);
    VTPrediction rightPrediction = vtPredict(rightX, modelMode, -1);
    bool phaseBoundary =
        modelMode == 1
        && leftPrediction.phaseTwo != rightPrediction.phaseTwo;
    int phaseOverride = -1;
    if (phaseBoundary)
    {
        bool useLeft =
            abs(leftPrediction.switchingMargin)
            >= abs(rightPrediction.switchingMargin);
        for (int column = 0; column < 5; ++column)
            faceX[column] = useLeft ? leftX[column] : rightX[column];
        phaseOverride = useLeft
            ? (leftPrediction.phaseTwo ? 1 : 0)
            : (rightPrediction.phaseTwo ? 1 : 0);
    }
    else if (modelMode == 1)
    {
        VTPrediction facePrediction = vtPredict(
            faceX, modelMode, -1
        );
        phaseOverride = facePrediction.phaseTwo ? 1 : 0;
    }

    VTEigensystem system = vtCarterSymbol(
        faceX, modelMode, phaseOverride
    );
    float maximumSpeed = 0.0;
    for (int index = 0; index < 25; ++index)
    {
        rightBasis[point, index] = system.right[index];
        leftBasis[point, index] = system.left[index];
    }
    for (int mode = 0; mode < 5; ++mode)
    {
        eigenvalues[point, mode] = system.values[mode];
        maximumSpeed = max(maximumSpeed, abs(system.values[mode]));
    }
    audit[point, 0] = maximumSpeed;
    audit[point, 1] = system.maximumResidual;
    audit[point, 2] = system.inverseResidual;
    audit[point, 3] = system.conditionEstimate;
    audit[point, 4] = system.minimumSeparation;
    audit[point, 5] = system.valid ? 1.0 : 0.0;
    audit[point, 6] = phaseBoundary ? 1.0 : 0.0;
    audit[point, 7] = 1.0 - maximumSpeed;
}

int vtPeriodicIndex(int index, int count)
{
    int wrapped = index % count;
    return wrapped < 0 ? wrapped + count : wrapped;
}

float vtWeno5Z(
    float v0,
    float v1,
    float v2,
    float v3,
    float v4
)
{
    float q0 = (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0;
    float q1 = (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0;
    float q2 = (2.0 * v2 + 5.0 * v3 - v4) / 6.0;
    float beta0 =
        (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) * (v0 - 2.0 * v1 + v2)
        + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2)
        * (v0 - 4.0 * v1 + 3.0 * v2);
    float beta1 =
        (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3)
        + 0.25 * (v1 - v3) * (v1 - v3);
    float beta2 =
        (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4)
        + 0.25 * (3.0 * v2 - 4.0 * v3 + v4)
        * (3.0 * v2 - 4.0 * v3 + v4);
    float tau5 = abs(beta0 - beta2);
    float scale = max(
        max(max(abs(v0), abs(v1)), max(abs(v2), abs(v3))),
        abs(v4)
    );
    float epsilon = 1.0e-26 + 1.0e-12 * scale * scale;
    float alpha0 =
        0.1 * (1.0 + pow(tau5 / (beta0 + epsilon), 2.0));
    float alpha1 =
        0.6 * (1.0 + pow(tau5 / (beta1 + epsilon), 2.0));
    float alpha2 =
        0.3 * (1.0 + pow(tau5 / (beta2 + epsilon), 2.0));
    return
        (alpha0 * q0 + alpha1 * q1 + alpha2 * q2)
        / (alpha0 + alpha1 + alpha2);
}

float vtProjectedSplit11(
    int face,
    int following,
    int sample,
    int mode,
    float signValue,
    float speed,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 2> rightBasis,
    ITensor<float, 2> leftBasis,
    ITensor<float, 2> entropyModes
)
{
    float physical[9] = {};
    for (int component = 0; component < 9; ++component)
    {
        physical[component] =
            flux[sample, component]
            + signValue * speed * state[sample, component];
    }
    if (mode < 9)
    {
        float value = 0.0;
        for (int component = 0; component < 9; ++component)
        {
            value +=
                leftBasis[face, mode * 9 + component]
                * physical[component];
        }
        return value;
    }
    float passive =
        flux[sample, mode]
        + signValue * speed * state[sample, mode];
    if (mode == 9)
    {
        for (int physicalMode = 0; physicalMode < 9; ++physicalMode)
        {
            float amplitude = 0.0;
            for (int component = 0; component < 9; ++component)
            {
                amplitude +=
                    leftBasis[
                        face, physicalMode * 9 + component
                    ]
                    * physical[component];
            }
            passive -=
                entropyModes[face, physicalMode] * amplitude;
        }
        return passive;
    }
    float leftTracer =
        state[face, 10] / max(abs(state[face, 0]), VT_EPSILON);
    float rightTracer =
        state[following, 10]
        / max(abs(state[following, 0]), VT_EPSILON);
    return
        passive
        - 0.5 * (leftTracer + rightTracer) * physical[0];
}

void theory33FullCharacteristicInterfaceKernel(
    int point,
    int axis,
    int nx,
    int ny,
    int nz,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 1> speed,
    ITensor<float, 2> rightBasis,
    ITensor<float, 2> leftBasis,
    ITensor<float, 2> entropyModes,
    IRWTensor<float, 2> interfaceFlux
)
{
    int im2 = vtGridNeighbor(point, axis, -2, nx, ny, nz);
    int im1 = vtGridNeighbor(point, axis, -1, nx, ny, nz);
    int ip1 = vtGridNeighbor(point, axis, 1, nx, ny, nz);
    int ip2 = vtGridNeighbor(point, axis, 2, nx, ny, nz);
    int ip3 = vtGridNeighbor(point, axis, 3, nx, ny, nz);
    float faceSpeed = max(speed[point], speed[ip1]) / 1.35;
    faceSpeed = clamp(faceSpeed, 1.0e-4, 1.0);
    float characteristic[11] = {};
    for (int mode = 0; mode < 11; ++mode)
    {
        float plus = vtWeno5Z(
            vtProjectedSplit11(
                point, ip1, im2, mode, 1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, im1, mode, 1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, point, mode, 1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, ip1, mode, 1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, ip2, mode, 1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            )
        );
        float minus = vtWeno5Z(
            vtProjectedSplit11(
                point, ip1, ip3, mode, -1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, ip2, mode, -1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, ip1, mode, -1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, point, mode, -1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            ),
            vtProjectedSplit11(
                point, ip1, im1, mode, -1.0, faceSpeed,
                state, flux, rightBasis, leftBasis, entropyModes
            )
        );
        characteristic[mode] = 0.5 * (plus + minus);
    }
    for (int component = 0; component < 9; ++component)
    {
        float value = 0.0;
        for (int mode = 0; mode < 9; ++mode)
        {
            value +=
                rightBasis[point, component * 9 + mode]
                * characteristic[mode];
        }
        interfaceFlux[point, component] = value;
    }
    float entropyValue = characteristic[9];
    for (int mode = 0; mode < 9; ++mode)
    {
        entropyValue +=
            entropyModes[point, mode] * characteristic[mode];
    }
    interfaceFlux[point, 9] = entropyValue;
    float leftTracer =
        state[point, 10] / max(abs(state[point, 0]), VT_EPSILON);
    float rightTracer =
        state[ip1, 10] / max(abs(state[ip1, 0]), VT_EPSILON);
    interfaceFlux[point, 10] =
        characteristic[10]
        + 0.5 * (leftTracer + rightTracer)
        * interfaceFlux[point, 0];
}

void theory33InterfaceKernel(
    int point,
    int reconstructionMode,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    IWTensor<float, 2> lowInterface,
    IWTensor<float, 2> highInterface
)
{
    int count = int(state.shape[0]);
    int im2 = vtPeriodicIndex(point - 2, count);
    int im1 = vtPeriodicIndex(point - 1, count);
    int ip1 = vtPeriodicIndex(point + 1, count);
    int ip2 = vtPeriodicIndex(point + 2, count);
    int ip3 = vtPeriodicIndex(point + 3, count);
    for (int column = 0; column < VT_STATE_WIDTH; ++column)
    {
        float low =
            0.5 * (flux[point, column] + flux[ip1, column])
            - 0.5 * (state[ip1, column] - state[point, column]);
        lowInterface[point, column] = low;
        if (reconstructionMode == 0)
        {
            highInterface[point, column] = low;
            continue;
        }
        float leftState = vtWeno5Z(
            state[im2, column],
            state[im1, column],
            state[point, column],
            state[ip1, column],
            state[ip2, column]
        );
        float rightState = vtWeno5Z(
            state[ip3, column],
            state[ip2, column],
            state[ip1, column],
            state[point, column],
            state[im1, column]
        );
        float leftFlux = vtWeno5Z(
            flux[im2, column],
            flux[im1, column],
            flux[point, column],
            flux[ip1, column],
            flux[ip2, column]
        );
        float rightFlux = vtWeno5Z(
            flux[ip3, column],
            flux[ip2, column],
            flux[ip1, column],
            flux[point, column],
            flux[im1, column]
        );
        highInterface[point, column] =
            0.5 * (leftFlux + rightFlux)
            - 0.5 * (rightState - leftState);
    }
}

float vtProjectedSplit(
    int face,
    int sample,
    int mode,
    float signValue,
    float speed,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 2> leftBasis
)
{
    float value = 0.0;
    for (int component = 0; component < 5; ++component)
    {
        value +=
            leftBasis[face, mode * 5 + component]
            * (
                flux[sample, component]
                + signValue * speed * state[sample, component]
            );
    }
    return value;
}

void theory33CharacteristicInterfaceKernel(
    int point,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 2> rightBasis,
    ITensor<float, 2> leftBasis,
    ITensor<float, 2> eigenvalues,
    IRWTensor<float, 2> highInterface
)
{
    int count = int(state.shape[0]);
    int im2 = vtPeriodicIndex(point - 2, count);
    int im1 = vtPeriodicIndex(point - 1, count);
    int ip1 = vtPeriodicIndex(point + 1, count);
    int ip2 = vtPeriodicIndex(point + 2, count);
    int ip3 = vtPeriodicIndex(point + 3, count);
    float speed = 0.0;
    for (int mode = 0; mode < 5; ++mode)
        speed = max(speed, abs(eigenvalues[point, mode]));
    speed = clamp(speed, 1.0e-4, 1.0);

    float characteristic[5] = {};
    for (int mode = 0; mode < 5; ++mode)
    {
        float plus = vtWeno5Z(
            vtProjectedSplit(
                point, im2, mode, 1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, im1, mode, 1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, point, mode, 1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, ip1, mode, 1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, ip2, mode, 1.0, speed,
                state, flux, leftBasis
            )
        );
        float minus = vtWeno5Z(
            vtProjectedSplit(
                point, ip3, mode, -1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, ip2, mode, -1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, ip1, mode, -1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, point, mode, -1.0, speed,
                state, flux, leftBasis
            ),
            vtProjectedSplit(
                point, im1, mode, -1.0, speed,
                state, flux, leftBasis
            )
        );
        characteristic[mode] = 0.5 * (plus + minus);
    }
    for (int component = 0; component < 5; ++component)
    {
        float value = 0.0;
        for (int mode = 0; mode < 5; ++mode)
        {
            value +=
                rightBasis[point, component * 5 + mode]
                * characteristic[mode];
        }
        highInterface[point, component] = value;
    }
}

void theory33CharacteristicRoeInterfaceKernel(
    int point,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 2> rightBasis,
    ITensor<float, 2> leftBasis,
    ITensor<float, 2> eigenvalues,
    IRWTensor<float, 2> highInterface
)
{
    int count = int(state.shape[0]);
    int following = point + 1 == count ? 0 : point + 1;
    float speed = 0.0;
    for (int mode = 0; mode < 5; ++mode)
        speed = max(speed, abs(eigenvalues[point, mode]));
    speed = clamp(speed, 1.0e-4, 1.0);

    float characteristicJump[5] = {};
    for (int mode = 0; mode < 5; ++mode)
    {
        for (int component = 0; component < 5; ++component)
        {
            characteristicJump[mode] +=
                leftBasis[point, mode * 5 + component]
                * (
                    state[following, component]
                    - state[point, component]
                );
        }
    }
    for (int component = 0; component < 5; ++component)
    {
        float dissipation = 0.0;
        for (int mode = 0; mode < 5; ++mode)
        {
            float rate =
                0.5 * abs(eigenvalues[point, mode])
                + 0.5 * speed;
            dissipation +=
                rightBasis[point, component * 5 + mode]
                * rate
                * characteristicJump[mode];
        }
        highInterface[point, component] =
            0.5 * (
                flux[point, component]
                + flux[following, component]
            )
            - 0.5 * dissipation;
    }

    // Entropy and tracer are passive extensions of the independent
    // five-field Carter block. Their characteristic speed is the material
    // transport speed; the zero-transport contact used by the parity
    // campaign reduces to the same half-Rusanov rate used by the CPU
    // extended Roe basis.
    for (int component = 5; component < VT_STATE_WIDTH; ++component)
    {
        highInterface[point, component] =
            0.5 * (
                flux[point, component]
                + flux[following, component]
            )
            - 0.25
            * speed
            * (
                state[following, component]
                - state[point, component]
            );
    }
}

void theory33InterfaceRhsKernel(
    int point,
    float inverseDx,
    ITensor<float, 2> interfaceFlux,
    ITensor<float, 2> pathData,
    IWTensor<float, 2> rhs
)
{
    int count = int(interfaceFlux.shape[0]);
    int previous = point == 0 ? count - 1 : point - 1;
    int following = point + 1 == count ? 0 : point + 1;
    float coefficientRight =
        0.5 * (pathData[point, 0] + pathData[following, 0]);
    float coefficientLeft =
        0.5 * (pathData[point, 0] + pathData[previous, 0]);
    float transportedRight =
        0.5 * (pathData[point, 2] + pathData[following, 2]);
    float transportedLeft =
        0.5 * (pathData[point, 2] + pathData[previous, 2]);
    float pathSource =
        coefficientRight
        * (pathData[following, 1] - pathData[point, 1])
        + coefficientLeft
        * (pathData[point, 1] - pathData[previous, 1])
        + transportedRight
        * (pathData[following, 3] - pathData[point, 3])
        + transportedLeft
        * (pathData[point, 3] - pathData[previous, 3]);
    for (int column = 0; column < VT_STATE_WIDTH; ++column)
    {
        rhs[point, column] =
            -(interfaceFlux[point, column] - interfaceFlux[previous, column])
            * inverseDx
            + (column == 4 ? 0.5 * inverseDx * pathSource : 0.0);
    }
}

float vtAdmissibility(
    float density,
    float momentum,
    float energy,
    float energyFloor
)
{
    return
        energy
        - sqrt(max(density * density + momentum * momentum, 0.0))
        - energyFloor;
}

void theory33PositivityThetaKernel(
    int point,
    float dt,
    float densityFloor,
    float carrierFloor,
    float energyFloor,
    ITensor<float, 2> state,
    ITensor<float, 2> lowRhs,
    ITensor<float, 2> highRhs,
    IWTensor<float, 1> theta
)
{
    float lowDensity = state[point, 0] + dt * lowRhs[point, 0];
    float highDensity = state[point, 0] + dt * highRhs[point, 0];
    float lowCarrier = state[point, 1] + dt * lowRhs[point, 1];
    float highCarrier = state[point, 1] + dt * highRhs[point, 1];
    float value = 1.0;
    if (highDensity < densityFloor)
    {
        value = min(
            value,
            clamp(
                (lowDensity - densityFloor)
                / max(lowDensity - highDensity, VT_EPSILON),
                0.0,
                1.0
            )
        );
    }
    if (highCarrier < carrierFloor)
    {
        value = min(
            value,
            clamp(
                (lowCarrier - carrierFloor)
                / max(lowCarrier - highCarrier, VT_EPSILON),
                0.0,
                1.0
            )
        );
    }
    float highMomentum = state[point, 2] + dt * highRhs[point, 2];
    float highEnergy = state[point, 3] + dt * highRhs[point, 3];
    if (
        vtAdmissibility(
            highDensity, highMomentum, highEnergy, energyFloor
        )
        < 0.0
    )
    {
        float lower = 0.0;
        float upper = value;
        [MaxIters(18)]
        for (int iteration = 0; iteration < 18; ++iteration)
        {
            float middle = 0.5 * (lower + upper);
            float density =
                lowDensity + middle * (highDensity - lowDensity);
            float momentum =
                state[point, 2]
                + dt
                * (
                    lowRhs[point, 2]
                    + middle
                    * (highRhs[point, 2] - lowRhs[point, 2])
                );
            float energy =
                state[point, 3]
                + dt
                * (
                    lowRhs[point, 3]
                    + middle
                    * (highRhs[point, 3] - lowRhs[point, 3])
                );
            if (
                vtAdmissibility(
                    density, momentum, energy, energyFloor
                )
                >= 0.0
            )
                lower = middle;
            else
                upper = middle;
        }
        value = lower;
    }
    theta[point] = value;
}

void theory33UnitThetaKernel(
    int point,
    IWTensor<float, 1> theta
)
{
    theta[point] = 1.0;
}

void theory33AccumulateThetaKernel(
    int point,
    ITensor<float, 1> stageTheta,
    IRWTensor<float, 1> minimumTheta
)
{
    minimumTheta[point] = min(
        minimumTheta[point], stageTheta[point]
    );
}

void theory33LimitedRhsKernel(
    int point,
    float inverseDx,
    ITensor<float, 2> lowInterface,
    ITensor<float, 2> highInterface,
    ITensor<float, 1> theta,
    ITensor<float, 2> pathData,
    IWTensor<float, 2> rhs
)
{
    int count = int(lowInterface.shape[0]);
    int previous = point == 0 ? count - 1 : point - 1;
    int following = point + 1 == count ? 0 : point + 1;
    float rightTheta = min(theta[point], theta[following]);
    float leftTheta = min(theta[previous], theta[point]);
    float coefficientRight =
        0.5 * (pathData[point, 0] + pathData[following, 0]);
    float coefficientLeft =
        0.5 * (pathData[point, 0] + pathData[previous, 0]);
    float transportedRight =
        0.5 * (pathData[point, 2] + pathData[following, 2]);
    float transportedLeft =
        0.5 * (pathData[point, 2] + pathData[previous, 2]);
    float pathSource =
        coefficientRight
        * (pathData[following, 1] - pathData[point, 1])
        + coefficientLeft
        * (pathData[point, 1] - pathData[previous, 1])
        + transportedRight
        * (pathData[following, 3] - pathData[point, 3])
        + transportedLeft
        * (pathData[point, 3] - pathData[previous, 3]);
    for (int column = 0; column < VT_STATE_WIDTH; ++column)
    {
        float right =
            lowInterface[point, column]
            + rightTheta
            * (
                highInterface[point, column]
                - lowInterface[point, column]
            );
        float left =
            lowInterface[previous, column]
            + leftTheta
            * (
                highInterface[previous, column]
                - lowInterface[previous, column]
            );
        rhs[point, column] =
            -(right - left) * inverseDx
            + (column == 4 ? 0.5 * inverseDx * pathSource : 0.0);
    }
}

void theory33RhsKernel(
    int point,
    float inverseDx,
    ITensor<float, 2> state,
    ITensor<float, 2> flux,
    ITensor<float, 2> pathData,
    IWTensor<float, 2> rhs
)
{
    int count = int(state.shape[0]);
    int previous = point == 0 ? count - 1 : point - 1;
    int following = point + 1 == count ? 0 : point + 1;
    float coefficientRight =
        0.5 * (pathData[point, 0] + pathData[following, 0]);
    float coefficientLeft =
        0.5 * (pathData[point, 0] + pathData[previous, 0]);
    float transportedRight =
        0.5 * (pathData[point, 2] + pathData[following, 2]);
    float transportedLeft =
        0.5 * (pathData[point, 2] + pathData[previous, 2]);
    float pathSource =
        coefficientRight
        * (pathData[following, 1] - pathData[point, 1])
        + coefficientLeft
        * (pathData[point, 1] - pathData[previous, 1])
        + transportedRight
        * (pathData[following, 3] - pathData[point, 3])
        + transportedLeft
        * (pathData[point, 3] - pathData[previous, 3]);
    for (int column = 0; column < VT_STATE_WIDTH; ++column)
    {
        float rightInterface =
            0.5 * (flux[point, column] + flux[following, column])
            - 0.5 * (state[following, column] - state[point, column]);
        float leftInterface =
            0.5 * (flux[previous, column] + flux[point, column])
            - 0.5 * (state[point, column] - state[previous, column]);
        rhs[point, column] =
            -(rightInterface - leftInterface) * inverseDx
            + (
                column == 4
                ? 0.5 * inverseDx * pathSource
                : 0.0
            );
    }
}

void theory33StageKernel(
    int point,
    float dt,
    float baseWeight,
    float evolvedWeight,
    ITensor<float, 2> baseState,
    ITensor<float, 2> stageState,
    ITensor<float, 2> rhs,
    IWTensor<float, 2> output
)
{
    for (int column = 0; column < VT_STATE_WIDTH; ++column)
    {
        output[point, column] =
            baseWeight * baseState[point, column]
            + evolvedWeight
            * (stageState[point, column] + dt * rhs[point, column]);
    }
}
