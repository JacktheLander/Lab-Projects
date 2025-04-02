module Rounder(PhaseOut, fraction, lookup1, lookup2, clk);
	input [15:0]PhaseOut;
	input clk;
	output reg [8:0]fraction;
	output reg [6:0]lookup1;
	output reg [6:0]lookup2;

	always@(*)begin
		fraction = PhaseOut[8:0];
		lookup1 = PhaseOut[15:9];
		lookup2 = lookup1+1;
	end

endmodule;
