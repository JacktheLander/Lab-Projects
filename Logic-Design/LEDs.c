#include "stm32l476xx.h"
#include "SysClock.h"
#include "stdint.h"
void red_on(){
	GPIOB->ODR |= GPIO_ODR_ODR_2;
	return;
}
	
void red_off(){
	GPIOB->ODR &= ~GPIO_ODR_ODR_2;
	return;
}
	
void red_toggle(){
	GPIOB->ODR ^= GPIO_ODR_ODR_2;
	return;
}

void green_on(){
	GPIOE->ODR |= GPIO_ODR_ODR_8;
	return;
}

void green_off(){
	GPIOE->ODR &= ~GPIO_ODR_ODR_8;
	return;
}

void green_toggle(){
	GPIOE->ODR ^= GPIO_ODR_ODR_8;
	return;
}

void delaymicro(unsigned int x){
		x=1;
		while(x!=6500000){
			x++;
		}
		return;
}

int main(void){

	System_Clock_Init(); // Switch System Clock = 80 MHz
	
	RCC->AHB2ENR |= 0x12;		//Port B and E Clock enabled
	
	GPIOB->MODER ^= GPIO_MODER_MODER2_1;		//Port B.2 set to push pull

	GPIOE->MODER ^= GPIO_MODER_MODER8_1;		//Port E.8 set to push pull
	
	//int i;
	//for(i=0;i<100;i++){
	//	red_toggle();
	//	delaymicro(500000);
	//}
	
	red_on();
	green_on();
	red_off();
	green_toggle();
	red_toggle();
	green_toggle();
	red_toggle();
	green_off();
	
	
	
	while(1) {
	red_toggle();
	delaymicro(500000);
	red_toggle();
	green_toggle();
	delaymicro(500000);
	green_toggle();
}
}

