//notes:







//Helper functions
void result_int(int ret1){
    asm ("");
}

void result_float(float ret1){
    asm ("");
}

void result_float_float(float ret1, float ret2){
    asm ("");
}


//Operations
void add_int_int(int arg1, int arg2){
    result_int(arg1 + arg2);
}

void add_float_float(float arg1, float arg2){
    result_float(arg1 + arg2);
}

void add_float_int(float arg1, float arg2){
    result_float(arg1 + arg2);
}

void sub_int_int(int arg1, int arg2){
    result_int(arg1 - arg2);
}

void sub_float_float(float arg1, float arg2){
    result_float(arg1 - arg2);
}

void sub_float_int(float arg1, int arg2){
    result_float(arg1 - arg2);
}

void sub_int_float(int arg1, float arg2){
    result_float(arg1 - arg2);
}

void mul_int_int(int arg1, int arg2){
    result_int(arg1 * arg2);
}

void mul_float_float(float arg1, float arg2){
    result_float(arg1 * arg2);
}

void mul_float_int(float arg1, int arg2){
    result_float(arg1 + arg2);
}

void div_int_int(int arg1, int arg2){
    result_int(arg1 / arg2);
}

void div_float_float(float arg1, float arg2){
    result_float(arg1 / arg2);
}

void div_float_int(float arg1, int arg2){
    result_float(arg1 / arg2);
}

void div_int_float(int arg1, float arg2){
    result_float(arg1 / arg2);
}

//Read global variables from heap
int read_int_ret = 1337;

void read_int(){
    result_int(read_int_ret);
}

float read_float_ret = 1337;

void read_float(){
    result_float(read_float_ret);
}

void read_float_2(float arg1){
    result_float_float(arg1, read_float_ret);
}