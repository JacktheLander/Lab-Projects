function [Distance] = HammingDist(msgBits, encBits)

difference = xor(msgBits, encBits);
Distance = 0;

for i = 1:length(difference)
    if difference(i) == 1
        Distance = Distance + 1;
    end
end
end
