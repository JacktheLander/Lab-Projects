#include <stdio.h>
#include <stdlib.h>

main(uint8_t integer, uint8_t decimal){
  int16_t i = integer;
  int8_t d = decimal >> 7;
  int16_t bits = (i << 8) + d;

  return bits;
}
  