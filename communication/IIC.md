[**Inter-Integrated Circuit**](https://www.ti.com/lit/an/sbaa565/sbaa565.pdf?ts=1794250662227&ref_url=https%253A%252F%252Fwww.google.com%252F)

Reference: [A Basic Guide to IIC](obsidian://open?vault=memento&file=communication%2Fresource%2FA%20Basic%20Guide%20to%20IIC.pdf)

I2C, often called I ‘two’ C, stands for the Inter-Integrated Circuit protocol (IIC).

![[Pasted image 20261110154944.png|621]]


+ SDA: a serial data line.
+ SCL: a serial clock line.
+ bidirectional, **half-duplex** communication.
+ support **multiple target devices** on a communication bus.
+ support **multiple controllers** that send and receive commands and data.
+ SDA, SCL have an **open-drain connection**, requiring a pullup resistor to a common voltage supply.

I2C was developed in 1982 by **Philips Semiconductor** (now NXP Semiconductor) as a low-speed communication protocol.

For connecting **controller devices** such as microcontrollers and processors **with target devices** such as data converters and other peripheral devices.


---

**I2C Speed Modes**

+ Standard-mode, **Sm**: Tops out at **100** kilobits per second (kbps).
+ Fast-mode, **Fm**: Tops out at **400** kilobits per second (kbps).
+ Fast-mode Plus, **Fm+**: Tops out at **1 megabit** per second (Mbps).
+ High-speed mode, **Hs-mode**: Tops out at **3.4 megabits** per second, the controller device must first use a controller code to allow for high-speed data transfer.
+ Ultra-Fast model, **UFm**: Tops out at **5 megabits** per second, write-only.

![[Pasted image 20261110205430.png|450]]


---

**Open-Drain Connection**

![[Pasted image 20261110210606.png]]

The open-drain connections are used on both SDA and SCL lines and connect to an NMOS transistor which controls the I2C communication line and pulls the line low. The open-drain refers to the NMOS bus connection when the NMOS is turned OFF. 


To set the voltage level of the SDA or SCL line, the NMOS is set on or off.

+ When the NMOS is on, the device pulls current through the resistor to ground. This pulls the open-drain line **low**. **A fast transition**, Figure 2-2.
+ When the NMOS turns off, the device stops pulling current, and the pullup resistor pulls the SDA or SCL line to VDD. The pullup resistor pulls the line **high**. **A slower transition**, Figure 2-3.

![[Pasted image 20261111233902.png|593]]

