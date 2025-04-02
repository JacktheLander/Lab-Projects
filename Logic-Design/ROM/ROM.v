module ROM(address, re, we, data, clk);
    input [9:0]address;
    input re, we;
    inout [11:0]data;
    
    reg [11:0]memory[1023:0];

    reg [11:0]out;

    assign data = (re) ? out : 8'bz;

    always @(posedge clk, we) begin
        if(we)
            memory[address] <= data;
    end

    always @(*) begin
        if(re)
            out = memory[address];
        else
            out = 8'bz;
    end
endmodule
