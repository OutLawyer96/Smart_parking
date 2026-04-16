package com.project.parkingbackend.service;

import com.project.parkingbackend.repository.VehicleRepository;
import org.springframework.stereotype.Service;

@Service
public class VehicleProfileService {

    private static final VehicleDimensions DEFAULT_DIMENSIONS = new VehicleDimensions(8.7, 3.6);

    private final VehicleRepository vehicleRepository;

    public VehicleProfileService(VehicleRepository vehicleRepository) {
        this.vehicleRepository = vehicleRepository;
    }

    public VehicleDimensions getDimensionsByPlate(String plateNumber) {
        if (plateNumber == null) return DEFAULT_DIMENSIONS;
        return vehicleRepository.findByPlateNumber(plateNumber)
                .map(v -> new VehicleDimensions(v.getLengthM(), v.getWidthM()))
                .orElse(DEFAULT_DIMENSIONS);
    }

    public record VehicleDimensions(double lengthM, double widthM) {}
}
