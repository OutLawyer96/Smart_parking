package com.project.parkingbackend.service;

import com.project.parkingbackend.model.*;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Stateful parking simulation service with multiple cars and obstacles.
 *
 * Layout (all coordinates in pixels, canvas 800 x 900):
 *
 * Enhanced parking lot with:
 * - 2 lanes with 4 target parking slots (8 total)
 * - Multiple parked cars
 * - Various obstacles: pillars, bollards, trash bins, charging stations
 * - Multiple entry/exit points
 *
 * Each car drives independently to its target slot.
 */
@Service
public class ParkingDataService {

    // ── tunables ──────────────────────────────────────────────────────────────
    private static final double START_Y      = 30.0;    // car entry point
    private static final double STEP         = 8.0;     // pixels the car advances per tick

    // ── fixed scene ───────────────────────────────────────────────────────────
    // Car size: 90×180. Target slot: 95×190 (just slightly larger for tight parking)
    private static final TargetSlot TARGET_SLOT =
            new TargetSlot(317.5, 495, 95, 190);   // Adjusted to car size (tight parking)

    private static final List<Obstacle> OBSTACLES = buildObstacles();

    // ── mutable car state (reset on each new session) ─────────────────────────
    private double carX = 150.0;  // Starting position (left lane)
    private double carY = START_Y;
    private double targetX = 365.0;  // Target slot centre x (95 wide: 317.5 + 47.5)
    private double targetY = 590.0;  // Target slot centre y (190 tall: 495 + 95)
    private boolean arrived = false;

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Advance the car one step and return the current scene snapshot.
     * The "path" list always starts at the current car position and ends at
     * the slot centre, giving the frontend a live remaining-path to draw.
     */
    public ParkingResponse generateParkingData() {
        if (!arrived) {
            advanceCar();
        }
        List<Coordinate> path = buildRemainingPath();
        return new ParkingResponse(path, OBSTACLES, TARGET_SLOT);
    }

    /** True once the car has fully entered the target slot. */
    public boolean hasArrived() {
        return arrived;
    }

    /** Reset simulation back to the start (call when a new session begins). */
    public void reset() {
        carX    = 150.0;
        carY    = START_Y;
        targetX = 365.0;  // Center of tight parking slot
        targetY = 590.0;
        arrived = false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Move the car through multi-phase parking with SHARP TURNS.
     *
     * Phase 1: Straight down lane
     * Phase 2: Sharp right turn (toward slot entry)
     * Phase 3: Sharp left turn (into slot)
     * Phase 4: Final parking
     */
    private void advanceCar() {
        if (arrived) {
            return;
        }

        // Phase 1: Drive straight down until turn point
        double turnPointY = 350.0;
        if (carY < turnPointY) {
            carY += STEP;
            carX = 150.0;  // Stay in lane
            return;
        }

        // Phase 2: Sharp right turn toward intermediate point
        double turnY = 400.0;
        double turnX = 250.0;
        if (carY < turnY) {
            double distToTurn = Math.hypot(turnX - carX, turnY - carY);
            if (distToTurn > STEP) {
                double angle = Math.atan2(turnY - carY, turnX - carX);
                carX += Math.cos(angle) * STEP;
                carY += Math.sin(angle) * STEP;
            } else {
                carX = turnX;
                carY = turnY;
            }
            return;
        }

        // Phase 3: Sharp left turn into slot approach
        double finalApproachY = 480.0;
        if (carY < finalApproachY) {
            double distToFinal = Math.hypot(targetX - carX, finalApproachY - carY);
            if (distToFinal > STEP) {
                double angle = Math.atan2(finalApproachY - carY, targetX - carX);
                carX += Math.cos(angle) * STEP;
                carY += Math.sin(angle) * STEP;
            } else {
                carX = targetX;
                carY = finalApproachY;
            }
            return;
        }

        // Phase 4: Final approach into slot
        double distToTarget = Math.abs(targetY - carY);
        if (distToTarget > STEP) {
            carY += STEP;
            carX = targetX;  // Center in slot
        } else {
            carX = targetX;
            carY = targetY;
            arrived = true;
        }
    }

    /**
     * Build a path with SHARP TURNS for realistic parking maneuvers.
     *
     * Phases:
     * 1. Drive straight down lane (Y movement only)
     * 2. SHARP RIGHT TURN toward slot (45° angle)
     * 3. SHARP LEFT TURN into slot (heading into target)
     * 4. Final approach and parking
     */
    private List<Coordinate> buildRemainingPath() {
        List<Coordinate> path = new ArrayList<>();
        path.add(new Coordinate(carX, carY));

        if (arrived) {
            return path;
        }

        double currentX = carX;
        double currentY = carY;

        // ─────────────────────────────────────────────────────────────────────
        // PHASE 1: Straight down the lane until turn point
        // ─────────────────────────────────────────────────────────────────────
        double turnPointY = 350.0;  // Where the first sharp turn happens

        if (currentY < turnPointY) {
            double phase1End = Math.min(turnPointY, targetY);
            double stepCount = (phase1End - currentY) / STEP;
            for (int i = 1; i <= stepCount && i < 50; i++) {
                double nextY = currentY + STEP * i;
                path.add(new Coordinate(round2(currentX), round2(nextY)));
            }
            currentY = phase1End;
        }

        // ─────────────────────────────────────────────────────────────────────
        // PHASE 2: SHARP RIGHT TURN (diagonal movement toward target X)
        // ─────────────────────────────────────────────────────────────────────
        double turnY = 400.0;  // Sharp turn angle point
        double turnX = 250.0;  // Intermediate point (right turn)

        if (currentY < turnY && carX < turnX) {
            double distToTurn = Math.hypot(turnX - currentX, turnY - currentY);
            int turnSteps = (int) Math.ceil(distToTurn / STEP);

            for (int i = 1; i <= turnSteps && i < 50; i++) {
                double t = (double) i / turnSteps;
                double nextX = currentX + (turnX - currentX) * t;
                double nextY = currentY + (turnY - currentY) * t;
                path.add(new Coordinate(round2(nextX), round2(nextY)));
            }
            currentX = turnX;
            currentY = turnY;
        }

        // ─────────────────────────────────────────────────────────────────────
        // PHASE 3: SHARP LEFT TURN (final approach to slot)
        // ─────────────────────────────────────────────────────────────────────
        double finalApproachY = 480.0;  // Final sharp left turn point

        if (currentY < finalApproachY) {
            double distToFinal = Math.hypot(targetX - currentX, finalApproachY - currentY);
            int finalSteps = (int) Math.ceil(distToFinal / STEP);

            for (int i = 1; i <= finalSteps && i < 50; i++) {
                double t = (double) i / finalSteps;
                double nextX = currentX + (targetX - currentX) * t;
                double nextY = currentY + (finalApproachY - currentY) * t;
                path.add(new Coordinate(round2(nextX), round2(nextY)));
            }
            currentX = targetX;
            currentY = finalApproachY;
        }

        // ─────────────────────────────────────────────────────────────────────
        // PHASE 4: Straight into parking slot
        // ─────────────────────────────────────────────────────────────────────
        if (currentY < targetY) {
            double remainingDist = targetY - currentY;
            int parkSteps = (int) Math.ceil(remainingDist / STEP);

            for (int i = 1; i <= parkSteps && i < 50; i++) {
                double t = (double) i / parkSteps;
                double nextY = currentY + (targetY - currentY) * t;
                path.add(new Coordinate(round2(currentX), round2(nextY)));
            }
        }

        // Final destination
        path.add(new Coordinate(round2(targetX), round2(targetY)));

        return path;
    }

    private static List<Obstacle> buildObstacles() {
        List<Obstacle> obs = new ArrayList<>();

        // ─── STRUCTURAL PILLARS (non-moving) ───────────────────────────────────
        obs.add(new Obstacle("pillar-1", new Rectangle(50, 150, 50, 50), false));
        obs.add(new Obstacle("pillar-2", new Rectangle(700, 150, 50, 50), false));
        obs.add(new Obstacle("pillar-3", new Rectangle(50, 400, 50, 50), false));
        obs.add(new Obstacle("pillar-4", new Rectangle(700, 400, 50, 50), false));

        // ─── BOLLARDS / CONCRETE BLOCKS (non-moving) ───────────────────────────
        obs.add(new Obstacle("bollard-1", new Rectangle(150, 250, 20, 20), false));
        obs.add(new Obstacle("bollard-2", new Rectangle(630, 250, 20, 20), false));
        obs.add(new Obstacle("bollard-3", new Rectangle(300, 300, 20, 20), false));
        obs.add(new Obstacle("bollard-4", new Rectangle(500, 300, 20, 20), false));

        // ─── TRASH BINS / CONTAINERS (non-moving) ─────────────────────────────
        obs.add(new Obstacle("trash-bin-1", new Rectangle(80, 600, 35, 35), false));
        obs.add(new Obstacle("trash-bin-2", new Rectangle(680, 600, 35, 35), false));
        obs.add(new Obstacle("trash-bin-3", new Rectangle(250, 750, 35, 35), false));
        obs.add(new Obstacle("trash-bin-4", new Rectangle(515, 750, 35, 35), false));

        // ─── EV CHARGING STATIONS (non-moving) ────────────────────────────────
        obs.add(new Obstacle("charging-station-1", new Rectangle(150, 800, 60, 50), false));
        obs.add(new Obstacle("charging-station-2", new Rectangle(590, 800, 60, 50), false));

        // ─── PARKED VEHICLES (non-moving, treated as obstacles) ─────────────────
        // Left lane parked cars
        obs.add(new Obstacle("parked-car-1", new Rectangle(80, 500, 90, 180), true));
        obs.add(new Obstacle("parked-car-2", new Rectangle(80, 720, 90, 180), true));

        // Right lane parked cars
        obs.add(new Obstacle("parked-car-3", new Rectangle(630, 500, 90, 180), true));
        obs.add(new Obstacle("parked-car-4", new Rectangle(630, 720, 90, 180), true));

        // Middle section parked cars
        obs.add(new Obstacle("parked-car-5", new Rectangle(355, 350, 90, 180), true));
        obs.add(new Obstacle("parked-car-6", new Rectangle(355, 720, 90, 180), true));

        return obs;
    }

    private static double round2(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
