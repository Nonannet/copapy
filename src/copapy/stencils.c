
    // Auto-generated stencils for copapy
    // Do not edit manually

    volatile int dummy_int = 1337;
    volatile float dummy_float = 1337;
    
    void result_int(int arg1);
    
    void result_float(float arg1);
    
    void result_int_int(int arg1, int arg2);
    
    void result_int_float(int arg1, float arg2);
    
    void result_float_int(float arg1, int arg2);
    
    void result_float_float(float arg1, float arg2);
    
    void add_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(arg1 + arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void add_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 + arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void add_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(arg1 + arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void add_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 + arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void sub_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(arg1 - arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void sub_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 - arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void sub_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(arg1 - arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void sub_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 - arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void mul_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(arg1 * arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void mul_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 * arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void mul_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(arg1 * arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void mul_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 * arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void div_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(arg1 / arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void div_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 / arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void div_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(arg1 / arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void div_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1 / arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg0_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(dummy_int, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg1_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(arg1, dummy_int);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg0_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(dummy_float, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg1_int_int(int arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_float(arg1, dummy_float);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg0_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_float(dummy_int, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg1_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(arg1, dummy_int);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg0_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(dummy_float, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg1_int_float(int arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_float(arg1, dummy_float);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg0_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_int(dummy_int, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg1_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(arg1, dummy_int);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg0_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(dummy_float, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg1_float_int(float arg1, int arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1, dummy_float);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg0_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_int_float(dummy_int, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_int_reg1_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_int(arg1, dummy_int);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg0_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(dummy_float, arg2);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void read_float_reg1_float_float(float arg1, float arg2) {
        asm volatile (".long 0xF17ECAFE");
        result_float_float(arg1, dummy_float);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void write_int(int arg1) {
        asm volatile (".long 0xF17ECAFE");
        dummy_int = arg1;
        result_int(arg1);
        asm volatile (".long 0xF27ECAFE");
    }
    
    void write_float(float arg1) {
        asm volatile (".long 0xF17ECAFE");
        dummy_float = arg1;
        result_float(arg1);
        asm volatile (".long 0xF27ECAFE");
    }
    
    int function_start(){
        result_int(0);  // dummy call instruction before marker gets striped
        asm volatile (".long 0xF27ECAFE");
        return 1;
    }
    
    int function_end(){
        result_int(0);
        asm volatile (".long 0xF17ECAFE");
        return 1;
    }
    