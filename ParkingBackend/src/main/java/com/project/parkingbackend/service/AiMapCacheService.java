package com.project.parkingbackend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.project.parkingbackend.config.AiIntegrationProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

@Service
public class AiMapCacheService {

    private static final Logger logger = LoggerFactory.getLogger(AiMapCacheService.class);

    private final AiHttpClient aiHttpClient;
    private final AiIntegrationProperties properties;
    private final AtomicReference<JsonNode> cachedMap = new AtomicReference<>();
    private final AtomicBoolean warmupStarted = new AtomicBoolean(false);

    public AiMapCacheService(AiHttpClient aiHttpClient, AiIntegrationProperties properties) {
        this.aiHttpClient = aiHttpClient;
        this.properties = properties;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void warmupCache() {
        if (!warmupStarted.compareAndSet(false, true)) {
            return;
        }

        Thread warmupThread = new Thread(this::pollUntilReady, "ai-map-warmup");
        warmupThread.setDaemon(true);
        warmupThread.start();
    }

    private void pollUntilReady() {
        logger.info("Waiting for AI map readiness...");

        while (true) {
            JsonNode envelope = aiHttpClient.fetchMapEnvelope();
            JsonNode mapNode = envelope == null ? null : envelope.get("map");

            if (mapNode != null && !mapNode.isNull()) {
                cachedMap.set(mapNode.deepCopy());
                logger.info("AI map ready and cached.");
                return;
            }

            if (!sleep(properties.getMapPollIntervalMs())) {
                return;
            }
        }
    }

    public JsonNode getCachedMap() {
        return cachedMap.get();
    }

    public boolean isMapReady() {
        return cachedMap.get() != null;
    }

    private boolean sleep(long delayMs) {
        try {
            Thread.sleep(delayMs);
            return true;
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            logger.warn("Map warmup interrupted");
            return false;
        }
    }
}




