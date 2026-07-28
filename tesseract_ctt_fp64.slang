// FP64 isolated conformal transverse-traceless initial-data solver.
//
// State order is psi, W^x, W^y, W^z.  The derivative buffer stores
// d_i psi, d_j W^i, and div W.  All edge closures match CartesianGrid(fd2).

int vtCTTCoordinate64(int point, int axis, int nx, int ny)
{
    return axis == 0
        ? point % nx
        : (axis == 1 ? (point / nx) % ny : point / (nx * ny));
}

int vtCTTCount64(int axis, int nx, int ny, int nz)
{
    return axis == 0 ? nx : (axis == 1 ? ny : nz);
}

int vtCTTNeighbor64(
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

double vtCTTInverse64(
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

bool vtCTTBoundary64(int point, int nx, int ny, int nz)
{
    int counts[3] = { nx, ny, nz };
    for (int axis = 0; axis < 3; ++axis)
    {
        int coordinate = vtCTTCoordinate64(point, axis, nx, ny);
        if (coordinate == 0 || coordinate == counts[axis] - 1)
            return true;
    }
    return false;
}

double vtCTTDerivative64(
    int point,
    int component,
    int axis,
    int nx,
    int ny,
    int nz,
    double inverseDx,
    double inverseDy,
    double inverseDz,
    StructuredBuffer<double> field
)
{
    int count = vtCTTCount64(axis, nx, ny, nz);
    int coordinate = vtCTTCoordinate64(point, axis, nx, ny);
    double inverse = vtCTTInverse64(
        axis, inverseDx, inverseDy, inverseDz
    );
    if (coordinate == 0)
    {
        int one = vtCTTNeighbor64(point, axis, 1, nx, ny, nz);
        int two = vtCTTNeighbor64(point, axis, 2, nx, ny, nz);
        return 0.5L * inverse * (
            -3.0L * field[component + 13 * point]
            + 4.0L * field[component + 13 * one]
            - field[component + 13 * two]
        );
    }
    if (coordinate == count - 1)
    {
        int one = vtCTTNeighbor64(point, axis, -1, nx, ny, nz);
        int two = vtCTTNeighbor64(point, axis, -2, nx, ny, nz);
        return 0.5L * inverse * (
            3.0L * field[component + 13 * point]
            - 4.0L * field[component + 13 * one]
            + field[component + 13 * two]
        );
    }
    int plus = vtCTTNeighbor64(point, axis, 1, nx, ny, nz);
    int minus = vtCTTNeighbor64(point, axis, -1, nx, ny, nz);
    return 0.5L * inverse * (
        field[component + 13 * plus]
        - field[component + 13 * minus]
    );
}

double vtCTTStateDerivative64(
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
    int count = vtCTTCount64(axis, nx, ny, nz);
    int coordinate = vtCTTCoordinate64(point, axis, nx, ny);
    double inverse = vtCTTInverse64(
        axis, inverseDx, inverseDy, inverseDz
    );
    if (coordinate == 0)
    {
        int one = vtCTTNeighbor64(point, axis, 1, nx, ny, nz);
        int two = vtCTTNeighbor64(point, axis, 2, nx, ny, nz);
        return 0.5L * inverse * (
            -3.0L * state[component + 4 * point]
            + 4.0L * state[component + 4 * one]
            - state[component + 4 * two]
        );
    }
    if (coordinate == count - 1)
    {
        int one = vtCTTNeighbor64(point, axis, -1, nx, ny, nz);
        int two = vtCTTNeighbor64(point, axis, -2, nx, ny, nz);
        return 0.5L * inverse * (
            3.0L * state[component + 4 * point]
            - 4.0L * state[component + 4 * one]
            + state[component + 4 * two]
        );
    }
    int plus = vtCTTNeighbor64(point, axis, 1, nx, ny, nz);
    int minus = vtCTTNeighbor64(point, axis, -1, nx, ny, nz);
    return 0.5L * inverse * (
        state[component + 4 * plus]
        - state[component + 4 * minus]
    );
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CTTInitialize64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    RWStructuredBuffer<double> state
)
{
    if (tid >= uint(numberPoints))
        return;
    state[4 * tid] = 1.0L;
    state[4 * tid + 1] = 0.0L;
    state[4 * tid + 2] = 0.0L;
    state[4 * tid + 3] = 0.0L;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CTTDerivatives64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float inverseDxInput,
    uniform float inverseDyInput,
    uniform float inverseDzInput,
    StructuredBuffer<double> state,
    RWStructuredBuffer<double> derivatives
)
{
    if (tid >= uint(numberPoints))
        return;
    double inverseDx = double(inverseDxInput);
    double inverseDy = double(inverseDyInput);
    double inverseDz = double(inverseDzInput);
    double divergence = 0.0L;
    for (int axis = 0; axis < 3; ++axis)
    {
        derivatives[13 * tid + axis] = vtCTTStateDerivative64(
            int(tid), 0, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz, state
        );
        for (int component = 0; component < 3; ++component)
        {
            double value = vtCTTStateDerivative64(
                int(tid), 1 + component, axis, nx, ny, nz,
                inverseDx, inverseDy, inverseDz, state
            );
            derivatives[13 * tid + 3 + 3 * component + axis] =
                value;
            if (component == axis)
                divergence += value;
        }
    }
    derivatives[13 * tid + 12] = divergence;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CTTResidual64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float inverseDxInput,
    uniform float inverseDyInput,
    uniform float inverseDzInput,
    uniform float kappaInput,
    StructuredBuffer<double> state,
    StructuredBuffer<double> source,
    StructuredBuffer<double> derivatives,
    RWStructuredBuffer<double> residual
)
{
    if (tid >= uint(numberPoints))
        return;
    double inverseDx = double(inverseDxInput);
    double inverseDy = double(inverseDyInput);
    double inverseDz = double(inverseDzInput);
    double kappa = double(kappaInput);
    double psi = max(state[4 * tid], 1.0e-30L);
    double laplacianPsi = 0.0L;
    double vectorLaplacian[3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        laplacianPsi += vtCTTDerivative64(
            int(tid), axis, axis, nx, ny, nz,
            inverseDx, inverseDy, inverseDz, derivatives
        );
        for (int component = 0; component < 3; ++component)
        {
            vectorLaplacian[component] += vtCTTDerivative64(
                int(tid), 3 + 3 * component + axis, axis,
                nx, ny, nz,
                inverseDx, inverseDy, inverseDz, derivatives
            );
        }
    }
    double divergence = derivatives[13 * tid + 12];
    double AbarSquared = 0.0L;
    for (int row = 0; row < 3; ++row)
    {
        vectorLaplacian[row] += (
            vtCTTDerivative64(
                int(tid), 12, row, nx, ny, nz,
                inverseDx, inverseDy, inverseDz, derivatives
            ) / 3.0L
        );
        for (int column = 0; column < 3; ++column)
        {
            double Abar =
                derivatives[13 * tid + 3 + 3 * column + row]
                + derivatives[13 * tid + 3 + 3 * row + column];
            if (row == column)
                Abar -= (2.0L / 3.0L) * divergence;
            AbarSquared += Abar * Abar;
        }
    }
    double psi2 = psi * psi;
    double psi4 = psi2 * psi2;
    double psi5 = psi4 * psi;
    double psi6 = psi5 * psi;
    double psi7 = psi6 * psi;
    residual[4 * tid] =
        laplacianPsi
        + 0.125L * AbarSquared / psi7
        + 0.25L * kappa * source[4 * tid] * psi5;
    for (int component = 0; component < 3; ++component)
    {
        residual[4 * tid + 1 + component] =
            vectorLaplacian[component]
            - kappa * psi6 * source[4 * tid + 1 + component];
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CTTUpdate64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    uniform float dtInput,
    uniform float minimumPsiInput,
    uniform float maximumPsiInput,
    StructuredBuffer<double> state,
    StructuredBuffer<double> residual,
    RWStructuredBuffer<double> candidate
)
{
    if (tid >= uint(numberPoints))
        return;
    if (vtCTTBoundary64(int(tid), nx, ny, nz))
    {
        candidate[4 * tid] = 1.0L;
        candidate[4 * tid + 1] = 0.0L;
        candidate[4 * tid + 2] = 0.0L;
        candidate[4 * tid + 3] = 0.0L;
        return;
    }
    double dt = double(dtInput);
    candidate[4 * tid] = clamp(
        state[4 * tid] + dt * residual[4 * tid],
        double(minimumPsiInput),
        double(maximumPsiInput)
    );
    for (int component = 1; component < 4; ++component)
    {
        candidate[4 * tid + component] =
            state[4 * tid + component]
            + dt * residual[4 * tid + component];
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void theory33CTTReduce64Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    uniform int nx,
    uniform int ny,
    uniform int nz,
    StructuredBuffer<double> state,
    StructuredBuffer<double> residual,
    RWStructuredBuffer<double> control
)
{
    if (tid != 0)
        return;
    double hSquared = 0.0L;
    double mSquared = 0.0L;
    double minimumPsi = 1.0e30L;
    double maximumPsi = -1.0e30L;
    int interiorCount = 0;
    for (int point = 0; point < numberPoints; ++point)
    {
        minimumPsi = min(minimumPsi, state[4 * point]);
        maximumPsi = max(maximumPsi, state[4 * point]);
        if (vtCTTBoundary64(point, nx, ny, nz))
            continue;
        hSquared += residual[4 * point] * residual[4 * point];
        for (int component = 1; component < 4; ++component)
        {
            mSquared +=
                residual[4 * point + component]
                * residual[4 * point + component];
        }
        interiorCount += 1;
    }
    double denominator = max(double(interiorCount), 1.0L);
    control[0] = sqrt(hSquared / denominator);
    control[1] = sqrt(mSquared / denominator);
    control[2] = minimumPsi;
    control[3] = maximumPsi;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void theory33CTTPackCCZ464Kernel(
    uint tid : SV_DispatchThreadID,
    uniform int numberPoints,
    StructuredBuffer<double> state,
    StructuredBuffer<double> derivatives,
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
    double divergence = derivatives[13 * tid + 12];
    double psi = max(state[4 * tid], 1.0e-30L);
    double psi2 = psi * psi;
    double psi6 = psi2 * psi2 * psi2;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            double Abar =
                derivatives[13 * tid + 3 + 3 * column + row]
                + derivatives[13 * tid + 3 + 3 * row + column];
            if (row == column)
                Abar -= (2.0L / 3.0L) * divergence;
            ccz4State[31 * tid + 9 + 3 * row + column] =
                float(Abar / psi6);
        }
    }
    ccz4State[31 * tid + 18] = float(1.0L / psi2);
    ccz4State[31 * tid + 24] = 1.0;
}
