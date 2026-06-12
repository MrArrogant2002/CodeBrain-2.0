
#include <stdio.h>

float read_voltage() {
    return 3.8;
}

void check_overvoltage(float voltage) {
    if(voltage > 4.2) {
        printf("Overvoltage detected\n");
    }
}
