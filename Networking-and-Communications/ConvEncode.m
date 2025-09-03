function [encoded] = ConvEncode(msgBits)

currentstate = 0;
output = [];

for i = 1:length(msgBits)
    
    input = msgBits(i);
    
    switch currentstate
        case 0
            if input == 1
                nextstate = 1;
                output = [output, 1, 1];
            else
                nextstate = 0;
                output = [output, 0, 0];
            end
        case 1
            if input == 1
                nextstate = 3;
                output = [output, 0, 0];
            else
                nextstate = 2;
                output = [output, 1, 1];
            end
        case 2
            if input == 1
                nextstate = 1;
                output = [output, 0, 1];
            else
                nextstate = 0;
                output = [output, 1, 0];
            end
        case 3
            if input == 1
                nextstate = 3;
                output = [output, 1, 0];
            else
                nextstate = 2;
                output = [output, 0, 1];
            end
    end
    
    currentstate = nextstate;
    encoded = output;
end
