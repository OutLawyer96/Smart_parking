package com.project.parkingbackend.repository;

import com.project.parkingbackend.model.Vehicle;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

@Repository
public class InMemoryVehicleRepository implements VehicleRepository {

    private static final double DEFAULT_LENGTH_M = 8.7;
    private static final double DEFAULT_WIDTH_M = 3.6;

    private final Map<String, Vehicle> store = new ConcurrentHashMap<>();

    public InMemoryVehicleRepository() {
        seed("DL8CAF5031", "Toyota",      "Camry");
        seed("MH12AB1234", "Honda",       "City");
        seed("KA01ZZ9999", "Ford",        "Endeavour");
        seed("TN09BC5678", "Maruti",      "Swift");
        seed("GJ05DE2345", "Hyundai",     "Creta");
        seed("UP32XY7890", "Tata",        "Nexon");
        seed("RJ14MN3456", "Volkswagen",  "Polo");
        seed("HR26PQ1122", "BMW",         "3 Series");
        seed("PB10RS4321", "Mercedes",    "C-Class");
        seed("WB20TU8888", "Kia",         "Seltos");
        seed("MH01AA0001", "Maruti",      "Baleno");
        seed("DL01CG1234", "Hyundai",     "i20");
    }

    private void seed(String plate, String make, String model) {
        store.put(plate.toUpperCase(), new Vehicle(plate, make, model, DEFAULT_LENGTH_M, DEFAULT_WIDTH_M));
    }

    @Override
    public Optional<Vehicle> findByPlateNumber(String plateNumber) {
        if (plateNumber == null || plateNumber.isBlank()) return Optional.empty();
        return Optional.ofNullable(store.get(plateNumber.trim().toUpperCase()));
    }

    @Override
    public List<Vehicle> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public void save(Vehicle vehicle) {
        if (vehicle == null || vehicle.getPlateNumber() == null) return;
        store.put(vehicle.getPlateNumber().toUpperCase(), vehicle);
    }

    @Override
    public boolean delete(String plateNumber) {
        if (plateNumber == null) return false;
        return store.remove(plateNumber.toUpperCase()) != null;
    }
}

