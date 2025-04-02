module lab7(PhaseIn, Load, rst, out, clk);
	input [15:0]PhaseIn;
	input Load, clk, rst;
	output [11:0]out;

	wire [15:0] Accumulator;

	Phase p(PhaseIn, Load, rst, Accumulator, clk);
	
	wire [6:0] lookup1, lookup2;
	wire [8:0] fraction;

	Rounder r(Accumulator, fraction, lookup1, lookup2, clk);

	reg [15:0] memory [0:127];
	reg [7:0] wave1;
	reg [7:0] wave2;

	always@(*) begin

		$readmemh("wavetable.txt", memory);

		wave1 = memory[lookup1][7:0];
		wave2 = memory[lookup2][7:0];
	end

	Interpolator interp(fraction, wave1, wave2, out, clk);

endmodule
