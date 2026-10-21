; x86_64_abi_shim.asm
; x86_64 assembly wrapper to preserve RSI and RDI registers when calling
; Copapy code compiled with System V ABI from runner compiled with Microsoft x64 ABI.
; In Microsoft x64 ABI, RSI and RDI are callee-saved. In System V ABI, they are caller-saved.
; This wrapper ensures RSI and RDI are preserved across the call.

.code

; extern int x86_64_abi_shim(entry_point_t entry_point);
; entry_point is passed in RCX (first argument in x64 Windows ABI)
; Returns int in EAX

x86_64_abi_shim PROC
    ; Save RSI and RDI (callee-saved in Microsoft x64 ABI)
    push rsi
    push rdi

    ; Allocate 40 bytes to maintain 16-byte stack alignment after call
    ; Enter with rsp = 16N, push 2 regs = 16N-16, sub 40 = 16N-56
    ; After call pushes return addr: 16N-64 = 16(N-4) aligned
    sub rsp, 40

    ; Call the entry point (function pointer is in RCX)
    mov rax, rcx
    call rax

    ; Restore stack - deallocate 40 bytes
    add rsp, 40

    ; Restore RSI and RDI
    pop rdi
    pop rsi

    ret
x86_64_abi_shim ENDP

end
