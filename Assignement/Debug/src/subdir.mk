################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CSR.cpp \
../src/Main.cpp 

C_SRCS += \
../src/testOpenmp.c 

O_SRCS += \
../src/CSR.o \
../src/Main.o \
../src/testOpenmp.o 

OBJS += \
./src/CSR.o \
./src/Main.o \
./src/testOpenmp.o 

CPP_DEPS += \
./src/CSR.d \
./src/Main.d 

C_DEPS += \
./src/testOpenmp.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


