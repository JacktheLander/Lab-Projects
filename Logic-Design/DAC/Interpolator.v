module Interpolator(fraction, wave1, wave2, out, clk);		//to interpolate - subtract wave1 from wave2, multiply the difference by fraction, add this to wave1
	input [8:0]fraction;
	input [7:0]wave1;
	input [7:0]wave2;
	input clk;
	output reg [11:0]out;
	reg [11:0] out_next;

	reg [7:0]sub;
	reg [11:0]frac;

	initial begin
		out = 12'd0;
		out_next = 12'd0;
	end

	always@(*) begin
		if(wave1>wave2) begin
			sub <= wave1-wave2;
			frac <= (sub*fraction)>>5;
			out_next[11:4] <= wave1;
			out <= out_next-frac-1;
		end
		else begin
			sub <= wave2-wave1;
			frac <= (sub*fraction)>>5;
			out_next[11:4] <= wave1;
			out <= out_next+frac;
		end
	end
endmodule
		

	
