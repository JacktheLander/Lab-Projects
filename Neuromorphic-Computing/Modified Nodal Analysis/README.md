
<!DOCTYPE html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Netlist and Matrices</title>
</head>
<body>

    <div class="container">
        <h2>Netlist</h2>
        <table>
            <tr>
                <th>DEV</th>
                <th>N1</th>
                <th>N2</th>
                <th>VAL</th>
            </tr>
            <tr><td>VS</td><td>1</td><td>0</td><td>1.000000</td></tr>
            <tr><td>R1</td><td>1</td><td>2</td><td>2000.000000</td></tr>
            <tr><td>R2</td><td>2</td><td>3</td><td>4000.000000</td></tr>
            <tr><td>C1</td><td>2</td><td>0</td><td>0.000008</td></tr>
            <tr><td>C2</td><td>3</td><td>0</td><td>0.000012</td></tr>
            <tr><td>C3</td><td>3</td><td>4</td><td>0.000010</td></tr>
            <tr><td>R3</td><td>0</td><td>4</td><td>5000.000000</td></tr>
        </table>

        <h2>Matrix A (Symbolic)</h2>
        <pre>
Matrix([
[ 1/R1,               -1/R1,                    0,            0, 1],
[-1/R1, C1/dt + 1/R2 + 1/R1,                -1/R2,            0, 0],
[    0,               -1/R2, C2/dt + C3/dt + 1/R2,       -C2/dt, 0],
[    0,                   0,               -C2/dt, C3/dt + 1/R3, 0],
[    1,                   0,                    0,            0, 0]])
        </pre>

        <h2>Vector X</h2>
        <pre>
Matrix([[V1, V2, V3, V4, IV1]])
        </pre>

        <h2>Vector Z</h2>
        <pre>
Matrix([[0, C1/dt*V2Prev, C2/dt*V3Prev + C3/dt*(V3Prev - V4Prev), C3/dt*(-V3Prev + V4Prev), VS]])
        </pre>

        <h2>Matrix A (Numerical)</h2>
        <table>
            <tr>
                <th>0</th><th>1</th><th>2</th><th>3</th><th>4</th>
            </tr>
            <tr><td>0.0005</td><td>-0.000500</td><td>0.000000</td><td>0.000000</td><td>1.0</td></tr>
            <tr><td>-0.0005</td><td>0.008746</td><td>-0.000250</td><td>0.000000</td><td>0.0</td></tr>
            <tr><td>0.0000</td><td>-0.000250</td><td>0.022239</td><td>-0.009995</td><td>0.0</td></tr>
            <tr><td>0.0000</td><td>0.000000</td><td>-0.009995</td><td>0.010195</td><td>0.0</td></tr>
            <tr><td>1.0000</td><td>0.000000</td><td>0.000000</td><td>0.000000</td><td>0.0</td></tr>
        </table>

        <h2>Vector Z</h2>
        <table>
            <tr><th>0</th></tr>
            <tr><td>0.0</td></tr>
            <tr><td>0.0</td></tr>
            <tr><td>0.0</td></tr>
            <tr><td>0.0</td></tr>
            <tr><td>1.0</td></tr>
        </table>

        <h2>Vector X</h2>
        <table>
            <tr><th>0</th></tr>
            <tr><td>1.000000</td></tr>
            <tr><td>0.057202</td></tr>
            <tr><td>0.001150</td></tr>
            <tr><td>0.001127</td></tr>
            <tr><td>-0.000471</td></tr>
        </table>

    </div>

</body>
</html>
