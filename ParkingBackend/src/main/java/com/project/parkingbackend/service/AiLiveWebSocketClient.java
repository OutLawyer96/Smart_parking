package com.project.parkingbackend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.parkingbackend.config.AiIntegrationProperties;
import com.project.parkingbackend.websocket.ParkingWebSocketHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.time.Duration;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

@Service
public class AiLiveWebSocketClient {

    private static final Logger logger = LoggerFactory.getLogger(AiLiveWebSocketClient.class);
    private static final Set<String> FORWARDED_EVENTS = Set.of("world_state", "parking_event", "slot_update");

    private final AiIntegrationProperties properties;
    private final ParkingWebSocketHandler frontendSocketHandler;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;
    private final ScheduledExecutorService reconnectExecutor;
    private final AtomicReference<WebSocket> upstreamSocket;
    private final AtomicInteger reconnectAttempt;
    private final AtomicBoolean reconnectScheduled;

    public AiLiveWebSocketClient(AiIntegrationProperties properties,
                                 ParkingWebSocketHandler frontendSocketHandler,
                                 ObjectMapper objectMapper) {
        this.properties = properties;
        this.frontendSocketHandler = frontendSocketHandler;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(properties.getHttpTimeoutMs()))
                .build();
        this.reconnectExecutor = Executors.newSingleThreadScheduledExecutor();
        this.upstreamSocket = new AtomicReference<>();
        this.reconnectAttempt = new AtomicInteger(0);
        this.reconnectScheduled = new AtomicBoolean(false);
    }

    @EventListener(ApplicationReadyEvent.class)
    public void connectOnStartup() {
        connect();
    }

    private void connect() {
        URI endpoint = URI.create(properties.getWebsocketUrl());
        logger.info("Connecting upstream AI websocket: {}", endpoint);

        CompletableFuture<WebSocket> future = httpClient.newWebSocketBuilder()
                .connectTimeout(Duration.ofMillis(properties.getHttpTimeoutMs()))
                .buildAsync(endpoint, new UpstreamListener());

        future.whenComplete((socket, error) -> {
            if (error != null) {
                frontendSocketHandler.setUpstreamConnected(false);
                logger.warn("AI websocket connect failed: {}", error.getMessage());
                scheduleReconnect();
                return;
            }
            upstreamSocket.set(socket);
            reconnectAttempt.set(0);
            reconnectScheduled.set(false);
            // setUpstreamConnected broadcasts "upstream_status connected" to all frontend clients
            frontendSocketHandler.setUpstreamConnected(true);
            logger.info("AI websocket connected to {}", properties.getWebsocketUrl());
        });
    }

    private void scheduleReconnect() {
        if (!reconnectScheduled.compareAndSet(false, true)) {
            return;
        }

        int attempt = reconnectAttempt.incrementAndGet();
        long expDelay = properties.getWsReconnectBaseDelayMs() * (1L << Math.min(20, attempt - 1));
        long delay = Math.min(expDelay, properties.getWsReconnectMaxDelayMs());

        reconnectExecutor.schedule(() -> {
            reconnectScheduled.set(false);
            connect();
        }, delay, TimeUnit.MILLISECONDS);
        logger.info("AI websocket reconnect attempt {} in {} ms", attempt, delay);
    }

    private class UpstreamListener implements WebSocket.Listener {

        private final StringBuilder messageBuffer = new StringBuilder();

        @Override
        public void onOpen(WebSocket webSocket) {
            webSocket.request(1);
            WebSocket.Listener.super.onOpen(webSocket);
        }

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            messageBuffer.append(data);

            if (last) {
                String payload = messageBuffer.toString();
                messageBuffer.setLength(0);
                forwardIfRelevant(payload);
            }

            webSocket.request(1);
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            frontendSocketHandler.setUpstreamConnected(false);
            logger.warn("AI websocket closed ({}): {}", statusCode, reason);
            scheduleReconnect();
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            frontendSocketHandler.setUpstreamConnected(false);
            logger.warn("AI websocket error: {}", error.getMessage());
            scheduleReconnect();
        }

        private void forwardIfRelevant(String payload) {
            try {
                JsonNode node = objectMapper.readTree(payload);
                String event = node.path("event").asText("");
                if (FORWARDED_EVENTS.contains(event)) {
                    frontendSocketHandler.broadcast(payload);
                }
            } catch (Exception ex) {
                logger.debug("Skipping non-JSON upstream message");
            }
        }
    }
}


