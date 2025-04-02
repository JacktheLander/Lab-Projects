module sram (address, re, we, data);

input [12:0]address;
input re, we;
inout [7:0]data;

reg [7:0]memory[8191:0];

reg [7:0]data_out;

assign data = (re) ? data_out : 8'bz; // Bidirectional data bus

always @(posedge we) begin
	memory[address] <= data;
end

always @(*) begin
	if(re)
		data_out = memory[address];
	else
		data_out = 8'bz;
end

endmodule
