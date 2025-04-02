module spitb();
    reg clk;
    reg rst;
    reg cs;
    reg sck;
    reg mosi;
    reg [7:0] data_to_send;
	reg [2:0] opcode;

    wire miso;
    wire [7:0] data_received;

    spi dut(rst, cs, sck, mosi, miso, data_received, data_to_send, opcode);

    always #5 clk = ~clk; // 100MHz clock

    reg [7:0] test_vector[0:3];
    reg [7:0] transfer_data;
    integer i, j;

    initial begin
        // Initialize Inputs
        clk = 0;
        rst = 1; // Apply reset initially
        cs = 1; // Chip select is inactive initially
        sck = 0;
        mosi = 0;
        data_to_send = 8'h00; // Initialize data to send as 0

        #20; rst = 0; // Release reset

        // Initialize test vectors
        test_vector[0] = 8'haa; // Test pattern 10101010
        test_vector[1] = 8'h55; // Test pattern 01010101
        test_vector[2] = 8'hff; // Test pattern 11111111
        test_vector[3] = 8'h00; // Test pattern 00000000

        // Wait for reset to complete
        #40;

        // Begin testing
        for (i=0; i<4; i=i+1) begin
            cs = 0; // Activate chip select
			opcode = 2'b00; // idle operation
			#10
			opcode = 2'b10; // write operation
            for (j=7; j>=0; j=j-1) begin
                mosi = test_vector[i][j];
                sck = 1;
                #10; // Full period for SCK high
                sck = 0;
            end

            // Check data_received with test_vector
            if (data_received !== test_vector[i]) begin
                $display("Test failed for vector %h, received %h", test_vector[i], data_received);
            end else begin
                $display("Test passed for vector %h", test_vector[i]);
            end

            // Prepare for read operation
            data_to_send = test_vector[i];

            // Start read operation
            cs = 0; // Activate chip select
			opcode = 2'b00;
			#10
			opcode = 2'b11;
            for (j=7; j>=0; j=j-1) begin
                sck = 1;
                #10; // Full period for SCK high
                sck = 0;
            end
			opcode = 2'b00; // idle operation
            cs = 1; // Deactivate chip select
            #20;

		end

        $finish; // End simulation
    end

endmodule
