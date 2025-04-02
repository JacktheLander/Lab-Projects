#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LENGTH 1024

FILE *outfile;

int main(int argc, char*argv[]){
    int out, address;
    outfile = fopen("data.txt", "w");
    printf("Start of data writing\n\n");
    for(int j = 0; j < LENGTH; j++){
        out = ((99*j)%1000);
		address = j*12;
        printf("row %d, address = %d, data = %d\n", j, address, out);
        fprintf(outfile, "we=1; re=0; address=10'd%d; data_in=12'd%d; //line %d\n#10\nwe=0; re=1; address=10'd%d;\n#10\n", address, out, j, address);
    }
    fclose(outfile);
    return 0;
}
