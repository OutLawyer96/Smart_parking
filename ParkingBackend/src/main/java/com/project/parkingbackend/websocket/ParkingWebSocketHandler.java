package com.project.parkingbackend.websocket;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.project.parkingbackend.model.PinSession;
import com.project.parkingbackend.service.AiMapCacheService;
import com.project.parkingbackend.service.PinSessionStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Frontend WebSocket fan-out handler.
 *
 * Upstream data-flow:
 *   ML model  →  ws://127.0.0.1:8000/ws/live  →  AiLiveWebSocketClient
 *                                                        ↓ broadcast()
 *   Browser   ←  ws://localhost:8080/ws/parking  ←  ParkingWebSocketHandler
 *
 * Two session modes:
 *  • Unsubscribed  – receives all events unfiltered (e.g. dashboard / admin view)
 *  • PIN-subscribed – receives world_state filtered to a single car + its
 *                     parking_event / slot_update events only
 */
@Component
public class ParkingWebSocketHandler extends TextWebSocketHandler {

    private static final Logger logger = LoggerFactory.getLogger(ParkingWebSocketHandler.class);

    // All active frontend sessions
    private static final Map<String, WebSocketSession> sessions = new ConcurrentHashMap<>();

    // sessionId → trackingId (only for PIN-subscribed sessions)
    private static final Map<String, Integer> sessionPinSubscription = new ConcurrentHashMap<>();

    private static volatile int connectedClients = 0;
    private static volatile boolean upstreamConnected = false;

    private final PinSessionStore pinSessionStore;
    private final AiMapCacheService mapCacheService;
    private final ObjectMapper objectMapper;

    public ParkingWebSocketHandler(PinSessionStore pinSessionStore,
                                   AiMapCacheService mapCacheService,
                                   ObjectMapper objectMapper) {
        this.pinSessionStore = pinSessionStore;
        this.mapCacheService = mapCacheService;
        this.objectMapper = objectMapper;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        sessions.put(session.getId(), session);
        connectedClients = sessions.size();
        logger.info("Frontend client connected [{}]. Total: {}", session.getId(), connectedClients);

        String ackStatus = upstreamConnected ? "connected" : "disconnected";
        safeSend(session,
                "{\"event\":\"connection_ack\"," +
                "\"status\":\"" + ackStatus + "\"," +
                "\"upstream_connected\":" + upstreamConnected + "," +
                "\"message\":\"Connected to Smart Parking backend. Send {\\\"action\\\":\\\"subscribe\\\",\\\"pin\\\":\\\"XXXX\\\"} to receive updates for your vehicle.\"}");
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        String raw = message.getPayload().trim();

        if ("ping".equalsIgnoreCase(raw)) {
            safeSend(session, "pong");
            return;
        }

        try {
            JsonNode node = objectMapper.readTree(raw);
            String action = node.path("action").asText("").toLowerCase();
            if ("subscribe".equals(action)) {
                handlePinSubscription(session, node.path("pin").asText("").trim());
            }
        } catch (Exception e) {
            logger.debug("Non-JSON or unknown message from [{}]: {}", session.getId(), raw);
        }
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        sessions.remove(session.getId());
        sessionPinSubscription.remove(session.getId());
        connectedClients = sessions.size();
        logger.info("Frontend client disconnected [{}]. Total: {}", session.getId(), connectedClients);
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable ex) {
        logger.error("Transport error [{}]: {}", session.getId(), ex.getMessage());
        sessions.remove(session.getId());
        sessionPinSubscription.remove(session.getId());
        connectedClients = sessions.size();
    }

    // ── PIN subscription ──────────────────────────────────────────────────────

    private void handlePinSubscription(WebSocketSession session, String pin) {
        if (pin.isBlank()) {
            safeSend(session, "{\"event\":\"error\",\"code\":\"MISSING_PIN\",\"message\":\"pin is required.\"}");
            return;
        }

        Optional<PinSession> opt = pinSessionStore.findByPin(pin);
        if (opt.isEmpty()) {
            safeSend(session, "{\"event\":\"error\",\"code\":\"INVALID_PIN\"," +
                    "\"message\":\"Invalid or expired PIN. Please try again.\"}");
            return;
        }

        PinSession ps = opt.get();
        sessionPinSubscription.put(session.getId(), ps.getTrackingId());
        logger.info("Session [{}] subscribed → tracking_id={} via PIN {}", session.getId(), ps.getTrackingId(), pin);

        try {
            ObjectNode ack = objectMapper.createObjectNode();
            ack.put("event", "subscription_ack");
            ack.put("tracking_id", ps.getTrackingId());
            ack.put("plate_number", ps.getPlate());
            ack.set("vehicle", objectMapper.valueToTree(ps.getVehicle()));
            ack.set("assigned_slot", objectMapper.valueToTree(ps.getAssignedSlot()));
            ack.set("route", objectMapper.valueToTree(ps.getRoute()));

            JsonNode map = mapCacheService.getCachedMap();
            ack.set("map", map != null ? map : objectMapper.nullNode());
            ack.put("message", "Subscribed. You will now receive live updates for your vehicle.");

            safeSend(session, objectMapper.writeValueAsString(ack));
        } catch (Exception e) {
            logger.error("Failed to build subscription_ack for [{}]: {}", session.getId(), e.getMessage());
        }
    }

    // ── Broadcast ─────────────────────────────────────────────────────────────

    /**
     * Push a JSON payload to every connected client.
     * Sessions with a PIN subscription receive a filtered version of world_state.
     */
    public void broadcast(String jsonPayload) {
        // Fast path: no PIN subscribers → just send to everyone as-is
        if (sessionPinSubscription.isEmpty()) {
            broadcastRaw(jsonPayload);
            return;
        }

        // Parse once for filtering
        JsonNode root = null;
        String eventType = null;
        try {
            root = objectMapper.readTree(jsonPayload);
            eventType = root.path("event").asText("");
        } catch (Exception e) {
            broadcastRaw(jsonPayload); // fallback
            return;
        }

        final JsonNode finalRoot = root;
        final String finalEvent = eventType;

        sessions.entrySet().removeIf(entry -> {
            String sid = entry.getKey();
            WebSocketSession s = entry.getValue();
            if (!s.isOpen()) {
                sessionPinSubscription.remove(sid);
                return true;
            }
            try {
                Integer subscribedId = sessionPinSubscription.get(sid);
                String payload = (subscribedId != null)
                        ? buildFiltered(finalRoot, finalEvent, subscribedId)
                        : jsonPayload;

                if (payload != null) {
                    synchronized (s) { s.sendMessage(new TextMessage(payload)); }
                }
                return false;
            } catch (IOException ex) {
                sessionPinSubscription.remove(sid);
                return true;
            }
        });
        connectedClients = sessions.size();
    }

    /** Send raw payload to all sessions with no filtering. */
    private void broadcastRaw(String payload) {
        sessions.entrySet().removeIf(entry -> {
            WebSocketSession s = entry.getValue();
            if (!s.isOpen()) { sessionPinSubscription.remove(entry.getKey()); return true; }
            try {
                synchronized (s) { s.sendMessage(new TextMessage(payload)); }
                return false;
            } catch (IOException ex) {
                sessionPinSubscription.remove(entry.getKey());
                return true;
            }
        });
        connectedClients = sessions.size();
    }

    /**
     * Build a filtered message for a PIN-subscribed session.
     *
     * world_state → keep only the car whose tracking_id matches
     * parking_event → forward only if tracking_id matches
     * slot_update → always forward (frontend decides relevance)
     * everything else → forward as-is
     *
     * Returns null to skip sending entirely.
     */
    private String buildFiltered(JsonNode root, String event, int trackingId) {
        try {
            return switch (event) {
                case "world_state" -> {
                    String ts = root.path("timestamp").asText();
                    JsonNode car = null;
                    for (JsonNode c : root.path("cars")) {
                        if (c.path("tracking_id").asInt() == trackingId) { car = c; break; }
                    }
                    ObjectNode out = objectMapper.createObjectNode();
                    out.put("event", "world_state");
                    out.put("timestamp", ts);
                    out.put("tracking_id", trackingId);
                    out.set("car", car != null ? car : objectMapper.nullNode());
                    yield objectMapper.writeValueAsString(out);
                }
                case "parking_event" ->
                        root.path("tracking_id").asInt() == trackingId
                                ? objectMapper.writeValueAsString(root)
                                : null;
                default -> objectMapper.writeValueAsString(root);
            };
        } catch (Exception e) {
            logger.debug("buildFiltered error: {}", e.getMessage());
            return null;
        }
    }

    // ── Upstream status ───────────────────────────────────────────────────────

    public void setUpstreamConnected(boolean connected) {
        upstreamConnected = connected;
        String status = connected ? "connected" : "disconnected";
        broadcast("{\"event\":\"upstream_status\",\"status\":\"" + status + "\"}");
        logger.info("Upstream AI WebSocket: {}", status);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void safeSend(WebSocketSession session, String text) {
        try {
            synchronized (session) { session.sendMessage(new TextMessage(text)); }
        } catch (IOException e) {
            logger.error("Could not send message to [{}]", session.getId());
        }
    }

    public static int getConnectedClientsCount() { return connectedClients; }
    public static boolean isStreamingActive()     { return upstreamConnected; }
}
