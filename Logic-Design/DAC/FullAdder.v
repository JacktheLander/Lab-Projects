module fulladder(inputx, inputy, ci, co, sum);
	input [15:0]inputx;
	input [15:0]inputy;
	input ci;
	output co;
	output [15:0]sum;
	
	wire c1;
    wire c2; 
    wire c3;
    
    adder4 a1 (x[3:0], y[3:0], ci, c1, sum[3:0]);
    adder4 a2 (x[7:4], y[7:4], c1, c2, sum[7:4]);
    adder4 a3 (x[11:8], y[11:8], c2, c3, sum[11:8]);
    adder4 a4 (x[15:12], y[15:12], c3, co, sum[15:12]);

endmodule

module adder4 (InputX, InputY, CI, CO, Sum);
	input [3:0] InputX, InputY;
	input CI;
	output CO;
	output [3:0] Sum;

	wire [2:0] c;

	full_adder f1 (Sum[0], c[0], CI, InputX[0], InputY[0]);
	full_adder f2 (Sum[1], c[1], c[0], InputX[1], InputY[1]);
	full_adder f3 (Sum[2], c[2], c[1], InputX[2], InputY[2]);
	full_adder f4 (Sum[3], CO, c[2], InputX[3], InputY[3]);

endmodule

module	full_adder(sum, cout, cin, inp1, inp2);

	input cin, inp1, inp2;
	output sum, cout;

	wire w1, w2, w3;
	xor x1(w1, inp1, inp2);
	xor x2(sum, w1, cin);
	and a1(w2, inp1, inp2);
	and a2(w3, w1, cin);
	or o1(cout, w2, w3);
endmodule
