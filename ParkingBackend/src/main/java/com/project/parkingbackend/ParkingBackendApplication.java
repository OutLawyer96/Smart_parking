package com.project.parkingbackend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class ParkingBackendApplication {

    public static void main(String[] args) {
        SpringApplication.run(ParkingBackendApplication.class, args);
    }

}
