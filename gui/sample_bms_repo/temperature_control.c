
#include <stdio.h>

float read_temperature() {
    return 55.0;
}

void check_temperature(float temp) {
    if(temp > 50) {
        printf("High temperature detected\n");
    }
}
