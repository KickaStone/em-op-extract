```shell
$ python -m torch.utils.collect_env
<frozen runpy>:128: RuntimeWarning: 'torch.utils.collect_env' found in sys.modules after import of package 'torch.utils', but prior to execution of 'torch.utils.collect_env'; this may result in unpredictable behaviour
Collecting environment information...
PyTorch version: 2.2.2
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Arch Linux (x86_64)
GCC version: (GCC) 13.2.1 20230801
Clang version: Could not collect
CMake version: version 3.29.0
Libc version: glibc-2.39

Python version: 3.12.2 | packaged by Anaconda, Inc. | (main, Feb 27 2024, 17:35:02) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.2-arch2-1-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.4.99
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3060 Ti
Nvidia driver version: 550.67
cuDNN version: Probably one of the following:
/usr/lib/libcudnn.so.8.9.7
/usr/lib/libcudnn_adv_infer.so.8.9.7
/usr/lib/libcudnn_adv_train.so.8.9.7
/usr/lib/libcudnn_cnn_infer.so.8.9.7
/usr/lib/libcudnn_cnn_train.so.8.9.7
/usr/lib/libcudnn_ops_infer.so.8.9.7
/usr/lib/libcudnn_ops_train.so.8.9.7
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               32
On-line CPU(s) list:                  0-31
Vendor ID:                            AuthenticAMD
Model name:                           AMD Ryzen 9 7950X 16-Core Processor
CPU family:                           25
Model:                                97
Thread(s) per core:                   2
Core(s) per socket:                   16
Socket(s):                            1
Stepping:                             2
CPU(s) scaling MHz:                   40%
CPU max MHz:                          5881.0000
CPU min MHz:                          400.0000
BogoMIPS:                             8986.33
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d
Virtualization:                       AMD-V
L1d cache:                            512 KiB (16 instances)
L1i cache:                            512 KiB (16 instances)
L2 cache:                             16 MiB (16 instances)
L3 cache:                             64 MiB (2 instances)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-31
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Mitigation; Safe RET
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS, IBPB conditional, STIBP always-on, RSB filling, PBRSB-eIBRS Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] torch==2.2.2
[pip3] torchaudio==2.2.2
[pip3] torchsummary==1.5.1
[pip3] torchvision==0.17.2
[conda] blas                      1.0                         mkl  
[conda] ffmpeg                    4.3                  hf484d3e_0    pytorch
[conda] libjpeg-turbo             2.0.0                h9bf148f_0    pytorch
[conda] mkl                       2023.1.0         h213fc3f_46344  
[conda] mkl-service               2.4.0           py312h5eee18b_1  
[conda] mkl_fft                   1.3.8           py312h5eee18b_0  
[conda] mkl_random                1.2.4           py312hdb19cb5_0  
[conda] numpy                     1.26.4          py312hc5e2394_0  
[conda] numpy-base                1.26.4          py312h0da6c21_0  
[conda] pytorch                   2.2.2           py3.12_cuda12.1_cudnn8.9.2_0    pytorch
[conda] pytorch-cuda              12.1                 ha16c6d3_5    pytorch
[conda] pytorch-mutex             1.0                        cuda    pytorch
[conda] torchaudio                2.2.2               py312_cu121    pytorch
[conda] torchsummary              1.5.1                    pypi_0    pypi
[conda] torchvision               0.17.2              py312_cu121    pytorch

```