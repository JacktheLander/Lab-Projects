module tb();
	reg [15:0]PhaseIn;
	reg Load, clk, rst;
	wire [11:0] out;
	reg [15:0] memory2 [0:199];

	lab7 dut(PhaseIn, Load, rst, out, clk);

	always begin
		#5 clk = ~clk;
		$display("clk = %b, rst = %b, PhaseIn = %d, out = %d, @ %d", clk, rst, PhaseIn, out, $time);
		#5 clk = ~clk;
	end
	
	integer j;
	initial begin
	$display("Start of test");
		clk<=0;	rst<=0;
		#10;
		rst<=1;
		#10
		rst<=0;
		for(j=0; j<200; j=j+1) begin
			memory2[j] <= out;
			PhaseIn <= 1; Load <= 1;
			#20; //2 cycle delay
		end
	$finish;
	end

	always@(*) begin

		$writememh("outputwave.txt", memory2);
	end

endmodule
