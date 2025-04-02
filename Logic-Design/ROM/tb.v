module tb();
	reg clk;
	reg re;
	reg we;
	wire [11:0]data;
	reg [9:0]address
	reg [11:0]datain;

	sram dut(address, re, we, data, clk);

	assign data = (we) ? data_in : 8'bz;

	always begin
		#5 clk = ~clk;
		%display("%d = 1023 - %d", address, data);		
		#5 clk = ~clk;
	end

	initial begin
		%display("Start of Test");
		clk<=0;
		#1
		we=1; re=0; address=10'd0; data_in=12'd0; //line 0
		#10
		we=0; re=1; address=10'd0;
		#10
		we=1; re=0; address=10'd12; data_in=12'd99; //line 1
		#10
		we=0; re=1; address=10'd12;
		#10
		we=1; re=0; address=10'd24; data_in=12'd198; //line 2
		#10
		we=0; re=1; address=10'd24;
		#10
		we=1; re=0; address=10'd36; data_in=12'd297; //line 3
		#10
		we=0; re=1; address=10'd36;
		#10
		we=1; re=0; address=10'd48; data_in=12'd396; //line 4
		#10
		we=0; re=1; address=10'd48;
		#10
		we=1; re=0; address=10'd60; data_in=12'd495; //line 5
		#10
		we=0; re=1; address=10'd60;
		#10
		we=1; re=0; address=10'd72; data_in=12'd594; //line 6
		#10
		we=0; re=1; address=10'd72;
		#10
		we=1; re=0; address=10'd84; data_in=12'd693; //line 7
		#10
		we=0; re=1; address=10'd84;
		#10
		we=1; re=0; address=10'd96; data_in=12'd792; //line 8
		#10
		we=0; re=1; address=10'd96;
		#10
		we=1; re=0; address=10'd108; data_in=12'd891; //line 9
		#10
		we=0; re=1; address=10'd108;
		#10
		we=1; re=0; address=10'd120; data_in=12'd990; //line 10
	$finish
	end
endmodule


