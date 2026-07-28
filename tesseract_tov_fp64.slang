// Single-workgroup fp64 Tolman-Oppenheimer-Volkoff profile generator.
//
// TOV integration is radially causal and therefore sequential.  It still
// executes on Vulkan: one invocation advances the RK4 radial recurrence while
// storing a profile that parallel grid-sampling kernels can consume.

double vtTOVDensity(double pressure, double K, double gamma)
{
    double ratio = max(pressure, 0.0L) / K;
    if (abs(gamma - 2.0L) < 1.0e-12L)
        return sqrt(ratio);
    return double(pow(float(ratio), float(1.0L / gamma)));
}

double vtTOVPressure(double density, double K, double gamma)
{
    if (abs(gamma - 2.0L) < 1.0e-12L)
        return K * density * density;
    return K * double(pow(float(density), float(gamma)));
}

double vtTOVLog(double value)
{
    return double(log(float(value)));
}

double vtTOVExp(double value)
{
    return double(exp(float(value)));
}

double vtTOVEnergy(double pressure, double K, double gamma)
{
    double safePressure = max(pressure, 0.0L);
    return vtTOVDensity(safePressure, K, gamma)
        + safePressure / (gamma - 1.0L);
}

void vtTOVDerivative(
    double radius,
    double mass,
    double pressure,
    double baryonicMass,
    double potential,
    double K,
    double gamma,
    out double result[4]
)
{
    pressure = max(pressure, 0.0L);
    double density = vtTOVDensity(pressure, K, gamma);
    double energy = vtTOVEnergy(pressure, K, gamma);
    double compact = max(
        radius * (radius - 2.0L * mass), 1.0e-300L
    );
    double gravity = (
        mass
        + 4.0L * 3.1415926535897932384626433832795L
        * radius * radius * radius * pressure
    ) / compact;
    double metricFactor = max(
        1.0L - 2.0L * mass / radius, 1.0e-14L
    );
    result[0] =
        4.0L * 3.1415926535897932384626433832795L
        * radius * radius * energy;
    result[1] = -(energy + pressure) * gravity;
    result[2] =
        4.0L * 3.1415926535897932384626433832795L
        * radius * radius * density / sqrt(metricFactor);
    result[3] = 2.0L * gravity;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void theory33TOVIntegrate64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int maximumSamples,
    uniform float centralDensityInput,
    uniform float KInput,
    uniform float gammaInput,
    uniform float drInput,
    uniform float pressureFloorFractionInput,
    uniform float maximumRadiusInput,
    RWStructuredBuffer<double> profile,
    RWStructuredBuffer<double> control
)
{
    if (tid != 0)
        return;
    double centralDensity = double(centralDensityInput);
    double K = double(KInput);
    double gamma = double(gammaInput);
    double dr = double(drInput);
    double pressureFloorFraction =
        double(pressureFloorFractionInput);
    double maximumRadius = double(maximumRadiusInput);
    double pi = 3.1415926535897932384626433832795L;
    double centralPressure =
        vtTOVPressure(centralDensity, K, gamma);
    double pressureFloor =
        pressureFloorFraction * centralPressure;
    double centralEnergy = vtTOVEnergy(
        centralPressure, K, gamma
    );
    double radius = 0.5L * dr;
    double mass =
        (4.0L * pi / 3.0L)
        * centralEnergy * radius * radius * radius;
    double baryonicMass =
        (4.0L * pi / 3.0L)
        * centralDensity * radius * radius * radius;
    double pressure = centralPressure;
    double potential = 0.0L;
    int count = 0;
    while (
        pressure > pressureFloor
        && radius < maximumRadius
        && radius > 2.0L * mass
        && count < maximumSamples
    )
    {
        double density = vtTOVDensity(pressure, K, gamma);
        double energy = vtTOVEnergy(pressure, K, gamma);
        int offset = 7 * count;
        profile[offset] = radius;
        profile[offset + 1] = density;
        profile[offset + 2] = pressure;
        profile[offset + 3] = energy;
        profile[offset + 4] = mass;
        profile[offset + 5] = 0.0L;
        profile[offset + 6] =
            1.0L / max(1.0L - 2.0L * mass / radius, 1.0e-14L);

        double y[4] = {
            mass, pressure, baryonicMass, potential
        };
        double k1[4] = {};
        double k2[4] = {};
        double k3[4] = {};
        double k4[4] = {};
        double stage[4] = {};
        vtTOVDerivative(
            radius, y[0], y[1], y[2], y[3],
            K, gamma, k1
        );
        for (int component = 0; component < 4; ++component)
            stage[component] =
                y[component] + 0.5L * dr * k1[component];
        vtTOVDerivative(
            radius + 0.5L * dr,
            stage[0], stage[1], stage[2], stage[3],
            K, gamma, k2
        );
        for (int component = 0; component < 4; ++component)
            stage[component] =
                y[component] + 0.5L * dr * k2[component];
        vtTOVDerivative(
            radius + 0.5L * dr,
            stage[0], stage[1], stage[2], stage[3],
            K, gamma, k3
        );
        for (int component = 0; component < 4; ++component)
            stage[component] =
                y[component] + dr * k3[component];
        vtTOVDerivative(
            radius + dr,
            stage[0], stage[1], stage[2], stage[3],
            K, gamma, k4
        );
        for (int component = 0; component < 4; ++component)
        {
            y[component] +=
                (dr / 6.0L)
                * (
                    k1[component]
                    + 2.0L * k2[component]
                    + 2.0L * k3[component]
                    + k4[component]
                );
        }
        mass = y[0];
        pressure = y[1];
        baryonicMass = y[2];
        potential = y[3];
        radius += dr;
        count += 1;
    }

    bool valid =
        count > 0
        && radius < maximumRadius
        && radius > 2.0L * mass
        && count < maximumSamples;
    if (!valid)
    {
        control[0] = double(count);
        control[4] = 0.0L;
        return;
    }
    double surfaceRadius = profile[7 * (count - 1)];
    double gravitationalMass = profile[7 * (count - 1) + 4];
    double nuSurfaceTarget = vtTOVLog(
        1.0L - 2.0L * gravitationalMass / surfaceRadius
    );
    double nu = 0.0L;
    profile[5] = 0.0L;
    for (int index = 1; index < count; ++index)
    {
        int previous = 7 * (index - 1);
        int current = 7 * index;
        double previousRadius = profile[previous];
        double currentRadius = profile[current];
        double previousGradient =
            2.0L
            * (
                profile[previous + 4]
                + 4.0L * pi
                * previousRadius * previousRadius * previousRadius
                * profile[previous + 2]
            )
            / (
                previousRadius
                * (previousRadius - 2.0L * profile[previous + 4])
            );
        double currentGradient =
            2.0L
            * (
                profile[current + 4]
                + 4.0L * pi
                * currentRadius * currentRadius * currentRadius
                * profile[current + 2]
            )
            / (
                currentRadius
                * (currentRadius - 2.0L * profile[current + 4])
            );
        nu +=
            0.5L
            * (previousGradient + currentGradient)
            * (currentRadius - previousRadius);
        profile[current + 5] = nu;
    }
    double shift = nuSurfaceTarget - nu;
    for (int index = 0; index < count; ++index)
    {
        int offset = 7 * index;
        profile[offset + 5] =
            vtTOVExp(0.5L * (profile[offset + 5] + shift));
    }
    control[0] = double(count);
    control[1] = surfaceRadius;
    control[2] = gravitationalMass;
    control[3] = baryonicMass;
    control[4] = 1.0L;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void theory33TOVInitialize64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform float centralDensityInput,
    uniform float KInput,
    uniform float gammaInput,
    uniform float drInput,
    RWStructuredBuffer<double> recurrence,
    RWStructuredBuffer<double> control
)
{
    if (tid != 0)
        return;
    double centralDensity = double(centralDensityInput);
    double K = double(KInput);
    double gamma = double(gammaInput);
    double dr = double(drInput);
    double centralPressure =
        vtTOVPressure(centralDensity, K, gamma);
    double centralEnergy = vtTOVEnergy(
        centralPressure, K, gamma
    );
    double radius = 0.5L * dr;
    double pi = 3.1415926535897932384626433832795L;
    recurrence[0] = radius;
    recurrence[1] =
        (4.0L * pi / 3.0L)
        * centralEnergy * radius * radius * radius;
    recurrence[2] =
        (4.0L * pi / 3.0L)
        * centralDensity * radius * radius * radius;
    recurrence[3] = centralPressure;
    recurrence[4] = 0.0L;
    recurrence[5] = 0.0L;
    recurrence[6] = 1.0L;
    recurrence[7] = 0.0L;
    recurrence[8] = 0.0L;
    for (int index = 0; index < 6; ++index)
        control[index] = 0.0L;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void theory33TOVAdvance64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int maximumSamples,
    uniform int batchSteps,
    uniform float centralDensityInput,
    uniform float KInput,
    uniform float gammaInput,
    uniform float drInput,
    uniform float pressureFloorFractionInput,
    uniform float maximumRadiusInput,
    RWStructuredBuffer<double> profile,
    RWStructuredBuffer<double> recurrence,
    RWStructuredBuffer<double> control
)
{
    if (tid != 0 || recurrence[6] < 0.5L)
        return;
    double centralDensity = double(centralDensityInput);
    double K = double(KInput);
    double gamma = double(gammaInput);
    double dr = double(drInput);
    double pressureFloorFraction =
        double(pressureFloorFractionInput);
    double maximumRadius = double(maximumRadiusInput);
    double centralPressure =
        vtTOVPressure(centralDensity, K, gamma);
    double pressureFloor =
        pressureFloorFraction * centralPressure;
    double pi = 3.1415926535897932384626433832795L;
    for (int localStep = 0; localStep < batchSteps; ++localStep)
    {
        double radius = recurrence[0];
        double mass = recurrence[1];
        double baryonicMass = recurrence[2];
        double pressure = recurrence[3];
        double potential = recurrence[4];
        int count = int(recurrence[5]);
        bool continueIntegration =
            pressure > pressureFloor
            && radius < maximumRadius
            && radius > 2.0L * mass
            && count < maximumSamples;
        if (!continueIntegration)
        {
            recurrence[6] = 0.0L;
            bool valid =
                count > 0
                && radius < maximumRadius
                && radius > 2.0L * mass
                && count < maximumSamples;
            control[0] = double(count);
            control[4] = valid ? 1.0L : 0.0L;
            if (count > 0)
            {
                control[1] = profile[7 * (count - 1)];
                control[2] = profile[7 * (count - 1) + 4];
                control[3] = baryonicMass;
                control[5] = recurrence[7];
            }
            break;
        }

        double density = vtTOVDensity(pressure, K, gamma);
        double energy = vtTOVEnergy(pressure, K, gamma);
        double gradient =
            2.0L
            * (
                mass
                + 4.0L * pi
                * radius * radius * radius * pressure
            )
            / (radius * (radius - 2.0L * mass));
        double cumulativeNu = recurrence[7];
        if (count > 0)
        {
            cumulativeNu +=
                0.5L
                * (recurrence[8] + gradient)
                * (radius - profile[7 * (count - 1)]);
        }
        int offset = 7 * count;
        profile[offset] = radius;
        profile[offset + 1] = density;
        profile[offset + 2] = pressure;
        profile[offset + 3] = energy;
        profile[offset + 4] = mass;
        profile[offset + 5] = cumulativeNu;
        profile[offset + 6] =
            1.0L / max(1.0L - 2.0L * mass / radius, 1.0e-14L);

        double y[4] = {
            mass, pressure, baryonicMass, potential
        };
        double k1[4] = {};
        double k2[4] = {};
        double k3[4] = {};
        double k4[4] = {};
        double stage[4] = {};
        vtTOVDerivative(
            radius, y[0], y[1], y[2], y[3],
            K, gamma, k1
        );
        for (int component = 0; component < 4; ++component)
            stage[component] =
                y[component] + 0.5L * dr * k1[component];
        vtTOVDerivative(
            radius + 0.5L * dr,
            stage[0], stage[1], stage[2], stage[3],
            K, gamma, k2
        );
        for (int component = 0; component < 4; ++component)
            stage[component] =
                y[component] + 0.5L * dr * k2[component];
        vtTOVDerivative(
            radius + 0.5L * dr,
            stage[0], stage[1], stage[2], stage[3],
            K, gamma, k3
        );
        for (int component = 0; component < 4; ++component)
            stage[component] =
                y[component] + dr * k3[component];
        vtTOVDerivative(
            radius + dr,
            stage[0], stage[1], stage[2], stage[3],
            K, gamma, k4
        );
        for (int component = 0; component < 4; ++component)
        {
            y[component] +=
                (dr / 6.0L)
                * (
                    k1[component]
                    + 2.0L * k2[component]
                    + 2.0L * k3[component]
                    + k4[component]
                );
        }
        recurrence[0] = radius + dr;
        recurrence[1] = y[0];
        recurrence[2] = y[2];
        recurrence[3] = y[1];
        recurrence[4] = y[3];
        recurrence[5] = double(count + 1);
        recurrence[7] = cumulativeNu;
        recurrence[8] = gradient;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33TOVFinalize64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int maximumSamples,
    RWStructuredBuffer<double> profile,
    StructuredBuffer<double> control
)
{
    int count = int(control[0]);
    if (
        tid >= uint(maximumSamples)
        || tid >= uint(count)
        || control[4] < 0.5L
    )
        return;
    double surfaceRadius = control[1];
    double gravitationalMass = control[2];
    double nuSurfaceTarget = vtTOVLog(
        1.0L - 2.0L * gravitationalMass / surfaceRadius
    );
    double shift = nuSurfaceTarget - control[5];
    profile[7 * tid + 5] =
        vtTOVExp(0.5L * (profile[7 * tid + 5] + shift));
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33TOVSampleCartesianKernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float dx,
    uniform float dy,
    uniform float dz,
    uniform float originX,
    uniform float originY,
    uniform float originZ,
    uniform float densityFloor,
    StructuredBuffer<double> profile,
    StructuredBuffer<double> control,
    RWStructuredBuffer<float> primitive,
    RWStructuredBuffer<float> ccz4State,
    RWStructuredBuffer<float> vectorState
)
{
    if (tid >= uint(numberPoints) || control[4] < 0.5L)
        return;
    int xIndex = int(tid) % nx;
    int yIndex = (int(tid) / nx) % ny;
    int zIndex = int(tid) / (nx * ny);
    float position[3] = {
        nx > 1
            ? originX + (float(xIndex) + 0.5) * dx
            : 0.0,
        ny > 1
            ? originY + (float(yIndex) + 0.5) * dy
            : 0.0,
        nz > 1
            ? originZ + (float(zIndex) + 0.5) * dz
            : 0.0
    };
    float radius = sqrt(
        position[0] * position[0]
        + position[1] * position[1]
        + position[2] * position[2]
    );
    int count = int(control[0]);
    double surfaceRadius = control[1];
    double gravitationalMass = control[2];
    double density;
    double lapse;
    double radialMetric;
    if (double(radius) <= surfaceRadius)
    {
        int lower = 0;
        int upper = count - 1;
        while (upper - lower > 1)
        {
            int middle = (lower + upper) / 2;
            if (profile[7 * middle] <= double(radius))
                lower = middle;
            else
                upper = middle;
        }
        double lowerRadius = profile[7 * lower];
        double upperRadius = profile[7 * upper];
        double weight = clamp(
            (double(radius) - lowerRadius)
            / max(upperRadius - lowerRadius, 1.0e-30L),
            0.0L,
            1.0L
        );
        density =
            (1.0L - weight) * profile[7 * lower + 1]
            + weight * profile[7 * upper + 1];
        lapse =
            (1.0L - weight) * profile[7 * lower + 5]
            + weight * profile[7 * upper + 5];
        radialMetric =
            (1.0L - weight) * profile[7 * lower + 6]
            + weight * profile[7 * upper + 6];
    }
    else
    {
        density = double(densityFloor);
        double compact = max(
            1.0L - 2.0L * gravitationalMass / double(radius),
            1.0e-8L
        );
        lapse = sqrt(compact);
        radialMetric = 1.0L / compact;
    }
    density = max(density, double(densityFloor));
    for (int component = 0; component < 9; ++component)
        primitive[9 * tid + component] = 0.0;
    primitive[9 * tid] = log(float(density));
    primitive[9 * tid + 1] = 0.0;
    primitive[9 * tid + 5] = log(float(densityFloor));
    for (int component = 0; component < 31; ++component)
        ccz4State[31 * tid + component] = 0.0;
    float normal[3] = {};
    if (radius > 1.0e-12)
    {
        normal[0] = position[0] / radius;
        normal[1] = position[1] / radius;
        normal[2] = position[2] / radius;
    }
    float radial = float(radialMetric);
    float conformalFactor = pow(max(radial, 1.0e-8), -1.0 / 6.0);
    float conformalScale = conformalFactor * conformalFactor;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            float physical =
                (row == column ? 1.0 : 0.0)
                + (radial - 1.0) * normal[row] * normal[column];
            ccz4State[31 * tid + 3 * row + column] =
                conformalScale * physical;
        }
    }
    ccz4State[31 * tid + 18] = conformalFactor;
    ccz4State[31 * tid + 24] = float(lapse);
    for (int component = 0; component < 16; ++component)
        vectorState[16 * tid + component] = 0.0;
}
