package com.project.parkingbackend.repository;

import com.project.parkingbackend.model.Vehicle;

import java.util.List;
import java.util.Optional;

public interface VehicleRepository {
    Optional<Vehicle> findByPlateNumber(String plateNumber);
    List<Vehicle> findAll();
    void save(Vehicle vehicle);
    boolean delete(String plateNumber);
}

