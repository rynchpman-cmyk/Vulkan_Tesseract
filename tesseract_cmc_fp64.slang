// Explicit multidimensional fp64 periodic CMC entry points.
//
// SlangPy's automatic ITensor<double> binder on Windows does not currently
// support fp64 tensors reliably on all Vulkan drivers, while raw
// StructuredBuffer<double> compute entry points do.  These kernels implement
// the complete one-, two-, and three-dimensional periodic CMC/Picard operator
// in fp64 without changing the float32 evolution ABI.

int vtCMCGridNeighbor64(
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
    // Avoid negative integer remainder here.  Some Windows Vulkan compiler
    // paths lower ``negative % positive`` inconsistently.  All stencil
    // offsets in this module are within [-2, 2], so two sign-free corrections
    // are sufficient even for a two-cell active axis.
    if (axis == 0)
    {
        x += offset;
        if (x < 0) x += nx;
        if (x < 0) x += nx;
        if (x >= nx) x -= nx;
        if (x >= nx) x -= nx;
    }
    else if (axis == 1)
    {
        y += offset;
        if (y < 0) y += ny;
        if (y < 0) y += ny;
        if (y >= ny) y -= ny;
        if (y >= ny) y -= ny;
    }
    else
    {
        z += offset;
        if (z < 0) z += nz;
        if (z < 0) z += nz;
        if (z >= nz) z -= nz;
        if (z >= nz) z -= nz;
    }
    return x + nx * (y + ny * z);
}

int vtCMCAxisCount64(int axis, int nx, int ny, int nz)
{
    return axis == 0 ? nx : (axis == 1 ? ny : nz);
}

double vtCMCInverseSpacing64(
    int axis,
    double inverseDx,
    double inverseDy,
    double inverseDz
)
{
    return axis == 0
        ? inverseDx
        : (axis == 1 ? inverseDy : inverseDz);
}

double vtCMCSquare64(double value)
{
    return value * value;
}

double vtCMCPower5_64(double value)
{
    double square = value * value;
    return square * square * value;
}

double vtCMCPower6_64(double value)
{
    return vtCMCPower5_64(value) * value;
}

double vtCMCPower7_64(double value)
{
    return vtCMCPower6_64(value) * value;
}

double vtCMCDerivative64(
    int point,
    int component,
    int axis,
    int nx,
    int ny,
    int nz,
    double inverseDx,
    double inverseDy,
    double inverseDz,
    StructuredBuffer<double> state
)
{
    if (vtCMCAxisCount64(axis, nx, ny, nz) <= 1)
        return 0.0L;
    int plus = vtCMCGridNeighbor64(
        point, axis, 1, nx, ny, nz
    );
    int minus = vtCMCGridNeighbor64(
        point, axis, -1, nx, ny, nz
    );
    return 0.5L
        * vtCMCInverseSpacing64(
            axis, inverseDx, inverseDy, inverseDz
        )
        * (
            state[4 * plus + component]
            - state[4 * minus + component]
        );
}

double vtCMCSecondDerivative64(
    int point,
    int component,
    int firstAxis,
    int secondAxis,
    int nx,
    int ny,
    int nz,
    double inverseDx,
    double inverseDy,
    double inverseDz,
    StructuredBuffer<double> state
)
{
    if (
        vtCMCAxisCount64(firstAxis, nx, ny, nz) <= 1
        || vtCMCAxisCount64(secondAxis, nx, ny, nz) <= 1
    )
        return 0.0L;
    double firstInverse = vtCMCInverseSpacing64(
        firstAxis, inverseDx, inverseDy, inverseDz
    );
    double secondInverse = vtCMCInverseSpacing64(
        secondAxis, inverseDx, inverseDy, inverseDz
    );
    if (firstAxis == secondAxis)
    {
        int plus = vtCMCGridNeighbor64(
            point, firstAxis, 2, nx, ny, nz
        );
        int minus = vtCMCGridNeighbor64(
            point, firstAxis, -2, nx, ny, nz
        );
        return 0.25L
            * firstInverse
            * firstInverse
            * (
                state[4 * plus + component]
                - 2.0L * state[4 * point + component]
                + state[4 * minus + component]
            );
    }
    int plusPlus = vtCMCGridNeighbor64(
        vtCMCGridNeighbor64(
            point, firstAxis, 1, nx, ny, nz
        ),
        secondAxis, 1, nx, ny, nz
    );
    int plusMinus = vtCMCGridNeighbor64(
        vtCMCGridNeighbor64(
            point, firstAxis, 1, nx, ny, nz
        ),
        secondAxis, -1, nx, ny, nz
    );
    int minusPlus = vtCMCGridNeighbor64(
        vtCMCGridNeighbor64(
            point, firstAxis, -1, nx, ny, nz
        ),
        secondAxis, 1, nx, ny, nz
    );
    int minusMinus = vtCMCGridNeighbor64(
        vtCMCGridNeighbor64(
            point, firstAxis, -1, nx, ny, nz
        ),
        secondAxis, -1, nx, ny, nz
    );
    return 0.25L
        * firstInverse
        * secondInverse
        * (
            state[4 * plusPlus + component]
            - state[4 * plusMinus + component]
            - state[4 * minusPlus + component]
            + state[4 * minusMinus + component]
        );
}

double vtCMCLaplacian64(
    int point,
    int component,
    int nx,
    int ny,
    int nz,
    double inverseDx,
    double inverseDy,
    double inverseDz,
    StructuredBuffer<double> state
)
{
    double result = 0.0L;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (vtCMCAxisCount64(axis, nx, ny, nz) <= 1)
            continue;
        int plus = vtCMCGridNeighbor64(
            point, axis, 1, nx, ny, nz
        );
        int minus = vtCMCGridNeighbor64(
            point, axis, -1, nx, ny, nz
        );
        double inverse = vtCMCInverseSpacing64(
            axis, inverseDx, inverseDy, inverseDz
        );
        result += inverse * inverse * (
            state[4 * plus + component]
            - 2.0L * state[4 * point + component]
            + state[4 * minus + component]
        );
    }
    return result;
}

void vtCMCAbar64(
    int point,
    int nx,
    int ny,
    int nz,
    double inverseDx,
    double inverseDy,
    double inverseDz,
    StructuredBuffer<double> state,
    out double Abar[9]
)
{
    double divergence = 0.0L;
    for (int axis = 0; axis < 3; ++axis)
    {
        divergence += vtCMCDerivative64(
            point, 1 + axis, axis,
            nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            state
        );
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            Abar[3 * row + column] =
                vtCMCDerivative64(
                    point, 1 + row, column,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    state
                )
                + vtCMCDerivative64(
                    point, 1 + column, row,
                    nx, ny, nz,
                    inverseDx, inverseDy, inverseDz,
                    state
                )
                - (
                    row == column
                    ? (2.0L / 3.0L) * divergence
                    : 0.0L
                );
        }
    }
}

double vtCMCAbarSquared64(
    int point,
    int nx,
    int ny,
    int nz,
    double inverseDx,
    double inverseDy,
    double inverseDz,
    StructuredBuffer<double> state
)
{
    double Abar[9] = {};
    vtCMCAbar64(
        point, nx, ny, nz,
        inverseDx, inverseDy, inverseDz,
        state, Abar
    );
    double result = 0.0L;
    for (int component = 0; component < 9; ++component)
        result += Abar[component] * Abar[component];
    return result;
}

void vtCMCResidual64(
    int point,
    int nx,
    int ny,
    int nz,
    double inverseDx,
    double inverseDy,
    double inverseDz,
    double kappaGravity,
    StructuredBuffer<double> state,
    StructuredBuffer<float> matterStress,
    StructuredBuffer<double> control,
    out double hamiltonian,
    out double momentum[3]
)
{
    double psi = max(state[4 * point], 1.0e-30L);
    double psi5 = vtCMCPower5_64(psi);
    hamiltonian =
        vtCMCLaplacian64(
            point, 0, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            state
        )
        + 0.125L
        * vtCMCAbarSquared64(
            point, nx, ny, nz,
            inverseDx, inverseDy, inverseDz,
            state
        )
        / vtCMCPower7_64(psi)
        - (control[0] / 12.0L) * psi5
        + 0.25L
        * kappaGravity
        * double(matterStress[13 * point])
        * psi5;
    for (int row = 0; row < 3; ++row)
    {
        double gradientDivergence = 0.0L;
        for (int inner = 0; inner < 3; ++inner)
        {
            gradientDivergence += vtCMCSecondDerivative64(
                point, 1 + inner, row, inner,
                nx, ny, nz,
                inverseDx, inverseDy, inverseDz,
                state
            );
        }
        momentum[row] =
            vtCMCLaplacian64(
                point, 1 + row,
                nx, ny, nz,
                inverseDx, inverseDy, inverseDz,
                state
            )
            + gradientDivergence / 3.0L
            - kappaGravity
            * vtCMCPower6_64(psi)
            * double(matterStress[13 * point + 1 + row]);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CMCTrace64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float inverseDx,
    uniform float inverseDy,
    uniform float inverseDz,
    uniform float kappaGravity,
    uniform int expanding,
    StructuredBuffer<double> state,
    StructuredBuffer<float> matterStress,
    RWStructuredBuffer<double> control
)
{
    if (tid != 0)
        return;
    double Aterm = 0.0L;
    double rhoTerm = 0.0L;
    double denominator = 0.0L;
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        double psi = max(state[4 * cell], 1.0e-30L);
        double psi5 = vtCMCPower5_64(psi);
        Aterm +=
            vtCMCAbarSquared64(
                cell, nx, ny, nz,
                double(inverseDx),
                double(inverseDy),
                double(inverseDz),
                state
            )
            / vtCMCPower7_64(psi);
        rhoTerm += double(matterStress[13 * cell]) * psi5;
        denominator += psi5;
    }
    double traceKSquared = (
        1.5L * Aterm + 3.0L * double(kappaGravity) * rhoTerm
    ) / max(denominator, 1.0e-30L);
    control[0] = max(traceKSquared, 0.0L);
    control[1] =
        (expanding != 0 ? -1.0L : 1.0L)
        * sqrt(max(traceKSquared, 0.0L));
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CMCUpdate64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float inverseDx,
    uniform float inverseDy,
    uniform float inverseDz,
    uniform float kappaGravity,
    uniform float pseudoTimeStep,
    uniform float minimumConformalFactor,
    uniform float maximumConformalFactor,
    StructuredBuffer<double> state,
    StructuredBuffer<float> matterStress,
    StructuredBuffer<double> control,
    RWStructuredBuffer<double> candidate
)
{
    if (tid >= uint(numberPoints))
        return;
    double hamiltonian;
    double momentum[3] = {};
    vtCMCResidual64(
        int(tid), nx, ny, nz,
        double(inverseDx),
        double(inverseDy),
        double(inverseDz),
        double(kappaGravity),
        state, matterStress, control,
        hamiltonian, momentum
    );
    candidate[4 * tid] = clamp(
        state[4 * tid] + double(pseudoTimeStep) * hamiltonian,
        double(minimumConformalFactor),
        double(maximumConformalFactor)
    );
    for (int axis = 0; axis < 3; ++axis)
    {
        candidate[4 * tid + 1 + axis] =
            state[4 * tid + 1 + axis]
            + double(pseudoTimeStep) * momentum[axis];
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CMCMean64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    StructuredBuffer<double> candidate,
    RWStructuredBuffer<double> control
)
{
    if (tid != 0)
        return;
    double sums[4] = {};
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        for (int component = 0; component < 4; ++component)
            sums[component] += candidate[4 * cell + component];
    }
    for (int component = 0; component < 4; ++component)
        control[2 + component] = sums[component] / numberPoints;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CMCNormalize64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    StructuredBuffer<double> candidate,
    StructuredBuffer<double> control,
    RWStructuredBuffer<double> state
)
{
    if (tid >= uint(numberPoints))
        return;
    state[4 * tid] =
        candidate[4 * tid] / max(control[2], 1.0e-30L);
    for (int axis = 0; axis < 3; ++axis)
    {
        state[4 * tid + 1 + axis] =
            candidate[4 * tid + 1 + axis] - control[3 + axis];
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CMCResidual64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float inverseDx,
    uniform float inverseDy,
    uniform float inverseDz,
    uniform float kappaGravity,
    StructuredBuffer<double> state,
    StructuredBuffer<float> matterStress,
    StructuredBuffer<double> control,
    RWStructuredBuffer<double> residual
)
{
    if (tid >= uint(numberPoints))
        return;
    double hamiltonian;
    double momentum[3] = {};
    vtCMCResidual64(
        int(tid), nx, ny, nz,
        double(inverseDx),
        double(inverseDy),
        double(inverseDz),
        double(kappaGravity),
        state, matterStress, control,
        hamiltonian, momentum
    );
    residual[2 * tid] = hamiltonian * hamiltonian;
    residual[2 * tid + 1] =
        momentum[0] * momentum[0]
        + momentum[1] * momentum[1]
        + momentum[2] * momentum[2];
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CMCReduce64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    StructuredBuffer<double> state,
    StructuredBuffer<double> residual,
    RWStructuredBuffer<double> control
)
{
    if (tid != 0)
        return;
    double hamiltonianSquared = 0.0L;
    double momentumSquared = 0.0L;
    double minimumPsi = 1.0e30L;
    double maximumPsi = -1.0e30L;
    for (int cell = 0; cell < numberPoints; ++cell)
    {
        hamiltonianSquared += residual[2 * cell];
        momentumSquared += residual[2 * cell + 1];
        minimumPsi = min(minimumPsi, state[4 * cell]);
        maximumPsi = max(maximumPsi, state[4 * cell]);
    }
    control[6] = sqrt(hamiltonianSquared / numberPoints);
    control[7] = sqrt(momentumSquared / numberPoints);
    control[8] = minimumPsi;
    control[9] = maximumPsi;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CMCPack64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float inverseDx,
    uniform float inverseDy,
    uniform float inverseDz,
    StructuredBuffer<double> state,
    StructuredBuffer<double> control,
    RWStructuredBuffer<float> ccz4State
)
{
    if (tid >= uint(numberPoints))
        return;
    for (int component = 0; component < 31; ++component)
        ccz4State[31 * tid + component] = 0.0;
    ccz4State[31 * tid] = 1.0;
    ccz4State[31 * tid + 4] = 1.0;
    ccz4State[31 * tid + 8] = 1.0;
    double Abar[9] = {};
    vtCMCAbar64(
        int(tid), nx, ny, nz,
        double(inverseDx),
        double(inverseDy),
        double(inverseDz),
        state, Abar
    );
    double psi = max(state[4 * tid], 1.0e-30L);
    double scaleA = 1.0L / vtCMCPower6_64(psi);
    for (int component = 0; component < 9; ++component)
    {
        ccz4State[31 * tid + 9 + component] =
            float(scaleA * Abar[component]);
    }
    ccz4State[31 * tid + 18] =
        float(1.0L / vtCMCSquare64(psi));
    ccz4State[31 * tid + 19] = float(control[1]);
    ccz4State[31 * tid + 24] = 1.0;
}
