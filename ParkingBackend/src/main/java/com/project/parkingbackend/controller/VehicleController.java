package com.project.parkingbackend.controller;

import com.project.parkingbackend.model.Vehicle;
import com.project.parkingbackend.repository.VehicleRepository;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/parking/vehicles")
public class VehicleController {

    private final VehicleRepository vehicleRepository;

    public VehicleController(VehicleRepository vehicleRepository) {
        this.vehicleRepository = vehicleRepository;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GET /api/parking/vehicles
    // List all registered vehicles in the database.
    // ─────────────────────────────────────────────────────────────────────────
    @GetMapping
    public ResponseEntity<List<Vehicle>> getAllVehicles() {
        return ResponseEntity.ok(vehicleRepository.findAll());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GET /api/parking/vehicles/{plateNumber}
    // Fetch a single vehicle by its registration plate.
    // ─────────────────────────────────────────────────────────────────────────
    @GetMapping("/{plateNumber}")
    public ResponseEntity<Object> getVehicle(@PathVariable String plateNumber) {
        Optional<Vehicle> vehicle = vehicleRepository.findByPlateNumber(plateNumber);
        if (vehicle.isPresent()) {
            return ResponseEntity.ok(vehicle.get());
        }
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(Map.of("message", "Vehicle not found: " + plateNumber.toUpperCase()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // POST /api/parking/vehicles
    // Add a new vehicle to the database.
    // ─────────────────────────────────────────────────────────────────────────
    @PostMapping
    public ResponseEntity<Object> addVehicle(@RequestBody Vehicle vehicle) {
        if (vehicle.getPlateNumber() == null || vehicle.getPlateNumber().isBlank()) {
            return ResponseEntity.badRequest()
                    .body(Map.of("message", "plate_number is required."));
        }
        if (vehicle.getLengthM() <= 0 || vehicle.getWidthM() <= 0) {
            return ResponseEntity.badRequest()
                    .body(Map.of("message", "length_m and width_m must be positive values."));
        }

        boolean exists = vehicleRepository.findByPlateNumber(vehicle.getPlateNumber()).isPresent();
        vehicleRepository.save(vehicle);
        HttpStatus status = exists ? HttpStatus.OK : HttpStatus.CREATED;
        return ResponseEntity.status(status).body(vehicle);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PUT /api/parking/vehicles/{plateNumber}
    // Update an existing vehicle's details.
    // ─────────────────────────────────────────────────────────────────────────
    @PutMapping("/{plateNumber}")
    public ResponseEntity<Object> updateVehicle(@PathVariable String plateNumber,
                                                @RequestBody Vehicle vehicle) {
        if (!vehicleRepository.findByPlateNumber(plateNumber).isPresent()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("message", "Vehicle not found: " + plateNumber.toUpperCase()));
        }
        vehicle.setPlateNumber(plateNumber);
        vehicleRepository.save(vehicle);
        return ResponseEntity.ok(vehicle);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DELETE /api/parking/vehicles/{plateNumber}
    // Remove a vehicle from the database.
    // ─────────────────────────────────────────────────────────────────────────
    @DeleteMapping("/{plateNumber}")
    public ResponseEntity<Object> deleteVehicle(@PathVariable String plateNumber) {
        boolean deleted = vehicleRepository.delete(plateNumber);
        if (deleted) {
            return ResponseEntity.ok(Map.of("message", "Vehicle " + plateNumber.toUpperCase() + " removed."));
        }
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(Map.of("message", "Vehicle not found: " + plateNumber.toUpperCase()));
    }
}

