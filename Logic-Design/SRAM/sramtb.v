module sramtb();

reg [12:0]address;
reg re, we;
wire [7:0]data;

reg [7:0]data_in;

sram dut(address, re, we, data);

assign data = (we) ? data_in : 8'bz; // Connect bidirectional data bus	

initial begin

        // Reset signals
        address = 0;
        we = 0;
        re = 0;
        data_in = 0;

        // Allow some time for initial conditions to settle
        #10;

        // Write some data to the SRAM
        address = 13'b0000000000001;
        data_in = 8'b10101010;
        we = 1;
        #10;
        we = 0;
        #10;

        address = 13'b0000000000010;
        data_in = 8'b11001100;
        we = 1;
        #10;
		data_in = 8'b11111111; //Test if the data only changes when we is asserted
		#10;
        we = 0;
        #10;
		data_in = 8'b01010101; 
		#10;		
		address = 13'b0000000000100; //Test if the data writes when we is low
		#10;

        // Read back the data and check
        address = 13'b0000000000001;
        re = 1;
        #10;
        $display("Read data at address 0000000000001: %b", data);
        re = 0;
        #10;

        address = 13'b0000000000010;
        re = 1;
        #10;
        $display("Read data at address 0000000000010: %b", data);
        re = 0;
        #10;
		
		//Test the output of the data when we and re are both low
		address = 13'b0000000000100;
		#10;
        $display("Read data at address 0000000000100: %b", data);		
		#10;

		//Test the last spot of the memory
		address = 13'b1111111111111;
		#10
		data_in = 8'b11111100;
		we = 1;
		#10
		re = 0;
		we = 1;
		#10
		$display("Read data at address 1111111111111: %b", data);
		#10;
			
        // End of test
        $finish;
end

endmodule
