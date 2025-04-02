module Phase(PhaseIn, Load, rst, Accumulator, clk);
	input [15:0]PhaseIn;
	input Load, clk, rst;
	reg [15:0]PhaseOut;
	reg [15:0]Phase;
	output reg [15:0]Accumulator;
	integer t;

	initial begin
		Accumulator = 16'h0000;
		Phase = 16'd0;
		t = 0;
	end

	always @(clk, Load) begin		
		t = t+1;
		if(rst) begin
			Accumulator <= 16'h0000;
		end
		
		if(Load) begin
			Phase <= (PhaseIn<<9)*128/100;
			if(t%4 == 0) begin
				Accumulator <= PhaseOut;
			end
		end



		PhaseOut = Phase + Accumulator;
	end
endmodule
