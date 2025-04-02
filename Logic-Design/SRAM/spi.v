module spi(rst, cs, sck, mosi, miso, data_received, data_to_send, opcode);
    input rst;         // Reset signal
    input cs;          // Chip Select (active low)
    input sck;         // SPI Clock
    input mosi;        // Master Out Slave In
    output reg miso;        // Master In Slave Out
    output reg [7:0] data_received; // Data received from master
    input [7:0] data_to_send;  // Data to send to master
	input [2:0] opcode;

    // State definitions
    parameter IDLE  = 2'b00;
    parameter WRITE = 2'b01;
    parameter READ  = 2'b10;

    // State and bit count variables
    reg [1:0] state = IDLE;    
    reg [2:0] bit_count = 0;

    always @(posedge sck or posedge rst) begin
        if (rst) begin
			state <= IDLE;
            bit_count <= 0;
            data_received <= 0;
            miso <= 1'bZ; 
        end
	end

    always @(posedge sck) begin
		if (!cs) begin
			if (opcode == 0) begin
				state <= IDLE;
				bit_count <= 0;
				miso <= 1'bZ;
			end else if (opcode == 2'b10) begin
				state <= WRITE;
                data_received <= (data_received << 1) | mosi;
                if (bit_count == 7) begin
                    state <= IDLE; // Move to IDLE state after WRITE is complete
                    bit_count <= 0; // Reset bit counter for READ operation
                end
                bit_count <= bit_count + 1;
			end else if (opcode == 2'b11) begin
				state <= READ;
                miso <= data_to_send[7 - bit_count]; // Send MSB first
                if (bit_count == 7) begin
                    state <= IDLE; // Return to IDLE after completing READ
                    bit_count <= 0;
					miso <= 1'bZ;
                end
                bit_count <= bit_count + 1;
			end			
		end else begin
            state <= IDLE;
            miso <= 1'bZ; // Ensure MISO is high impedance when not in use
            bit_count <= 0;
		end
	end
endmodule
