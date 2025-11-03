# Crash on windows

## Issue

Calling the function "entr_point" leads to an crash of the calling executable
("Return value: 1" is the last output) if the caller is compiled with MSVC and
optimization (/O2). For MSVC without optimization and gcc with -O3 there is no issue.

## Callee

The callee code is composed of stencils build with gcc-12 -O3:

```
    1000:       f3 0f 1e fa             endbr64
    1004:       48 83 ec 08             sub    $0x8,%rsp
    1008:       31 ff                   xor    %edi,%edi
    100a:       8b 3d fc 1f 00 00       mov    0x1ffc(%rip),%edi        # 0x300c
    1010:       8b 35 fa 1f 00 00       mov    0x1ffa(%rip),%esi        # 0x3010
    1016:       0f af fe                imul   %esi,%edi
    1019:       8b 35 e5 1f 00 00       mov    0x1fe5(%rip),%esi        # 0x3004
    101f:       01 f7                   add    %esi,%edi
    1021:       89 3d ed 1f 00 00       mov    %edi,0x1fed(%rip)        # 0x3014
    1027:       8b 3d d3 1f 00 00       mov    0x1fd3(%rip),%edi        # 0x3000
    102d:       8b 35 e5 1f 00 00       mov    0x1fe5(%rip),%esi        # 0x3018
    1033:       0f af fe                imul   %esi,%edi
    1036:       89 3d e0 1f 00 00       mov    %edi,0x1fe0(%rip)        # 0x301c
    103c:       8b 3d d2 1f 00 00       mov    0x1fd2(%rip),%edi        # 0x3014
    1042:       8b 35 d4 1f 00 00       mov    0x1fd4(%rip),%esi        # 0x301c
    1048:       01 f7                   add    %esi,%edi
    104a:       89 3d b8 1f 00 00       mov    %edi,0x1fb8(%rip)        # 0x3008
    1050:       b8 01 00 00 00          mov    $0x1,%eax
    1055:       48 83 c4 08             add    $0x8,%rsp
    1059:       c3                      retq
```

## Caller

The caller (src/copapy/runmem.c) calls the callee (here named "entr_point") like so:

``` c
case RUN_PROG:
    rel_entr_point = *(uint32_t*)bytes; bytes += 4;
    printf("RUN_PROG rel_entr_point=%i\n", rel_entr_point);
    entr_point = (int (*)())(executable_memory + rel_entr_point);  

    mark_mem_executable(executable_memory, executable_memory_len);
    int ret = entr_point();
    printf("Return value: %i\n", ret);
    break;
```

Compiled with MSVC and optimization (crash):

```
  00000001400078CA: 8B 1E              mov         ebx,dword ptr [rsi]
  00000001400078CC: 48 8D 0D AD 30 08  lea         rcx,[??_C@_0BM@DFBMBFEB@RUN_PROG?5rel_entr_point?$DN?$CFi?6@]
                    00
  00000001400078D3: 8B D3              mov         edx,ebx
  00000001400078D5: 48 83 C6 04        add         rsi,4
  00000001400078D9: E8 34 AD FF FF     call        @ILT+5645(printf)
  00000001400078DE: 48 8B 0D 0B B9 09  mov         rcx,qword ptr [executable_memory]
                    00
  00000001400078E5: 8B 15 01 B9 09 00  mov         edx,dword ptr [executable_memory_len]
  00000001400078EB: 48 8D 04 19        lea         rax,[rcx+rbx]
  00000001400078EF: 48 89 05 0A B9 09  mov         qword ptr [entr_point],rax
                    00
  00000001400078F6: E8 A8 B7 FF FF     call        @ILT+8350(mark_mem_executable)
  00000001400078FB: FF 15 FF B8 09 00  call        qword ptr [entr_point]
  0000000140007901: 8B D0              mov         edx,eax
  0000000140007903: 48 8D 0D A6 27 08  lea         rcx,[??_C@_0BC@LGACBIJC@Return?5value?3?5?$CFi?6@]
                    00
  000000014000790A: E8 03 AD FF FF     call        @ILT+5645(printf)
  000000014000790F: E9 A8 00 00 00     jmp         00000001400079BC
```

Compiled with MSVC and no optimization (no crash):

```
  0000000140007978: 48 8B 44 24 70     mov         rax,qword ptr [rsp+70h]
  000000014000797D: 8B 00              mov         eax,dword ptr [rax]
  000000014000797F: 89 44 24 4C        mov         dword ptr [rsp+4Ch],eax
  0000000140007983: 48 8B 44 24 70     mov         rax,qword ptr [rsp+70h]
  0000000140007988: 48 83 C0 04        add         rax,4
  000000014000798C: 48 89 44 24 70     mov         qword ptr [rsp+70h],rax
  0000000140007991: 8B 54 24 4C        mov         edx,dword ptr [rsp+4Ch]
  0000000140007995: 48 8D 0D B4 97 09  lea         rcx,[1400A1150h]
                    00
  000000014000799C: E8 71 AC FF FF     call        @ILT+5645(printf)
  00000001400079A1: 8B 44 24 4C        mov         eax,dword ptr [rsp+4Ch]
  00000001400079A5: 48 8B 0D 94 AB 09  mov         rcx,qword ptr [executable_memory]
                    00
  00000001400079AC: 48 03 C8           add         rcx,rax
  00000001400079AF: 48 8B C1           mov         rax,rcx
  00000001400079B2: 48 89 05 97 AB 09  mov         qword ptr [entr_point],rax
                    00
  00000001400079B9: 8B 15 7D AB 09 00  mov         edx,dword ptr [executable_memory_len]
  00000001400079BF: 48 8B 0D 7A AB 09  mov         rcx,qword ptr [executable_memory]
                    00
  00000001400079C6: E8 D8 B6 FF FF     call        @ILT+8350(mark_mem_executable)
  00000001400079CB: FF 15 7F AB 09 00  call        qword ptr [entr_point]
  00000001400079D1: 89 44 24 54        mov         dword ptr [rsp+54h],eax
  00000001400079D5: 8B 54 24 54        mov         edx,dword ptr [rsp+54h]
  00000001400079D9: 48 8D 0D 90 97 09  lea         rcx,[1400A1170h]
                    00
  00000001400079E0: E8 2D AC FF FF     call        @ILT+5645(printf)
```