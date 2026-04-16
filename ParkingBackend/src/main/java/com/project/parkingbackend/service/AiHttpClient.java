package com.project.parkingbackend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.parkingbackend.config.AiIntegrationProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

@Service
public class AiHttpClient {

    private static final Logger logger = LoggerFactory.getLogger(AiHttpClient.class);

    private final AiIntegrationProperties properties;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;

    public AiHttpClient(AiIntegrationProperties properties, ObjectMapper objectMapper) {
        this.properties = properties;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(properties.getHttpTimeoutMs()))
                .build();
    }

    public JsonNode fetchMapEnvelope() {
        return fetchMapResponse().body();
    }

    public JsonNode fetchSlotsEnvelope() {
        return fetchSlotsResponse().body();
    }

    public AiHttpResponse fetchMapResponse() {
        return sendGet("/api/v1/map");
    }

    public AiHttpResponse fetchSlotsResponse() {
        return sendGet("/api/v1/slots");
    }

    public AiHttpResponse gateMatchWithRetry(int trackingId, double lengthM, double widthM) {
        int attempts = Math.max(1, properties.getGateMatchMaxRetries() + 1);
        AiHttpResponse lastResponse = null;

        for (int i = 0; i < attempts; i++) {
            lastResponse = sendGateMatch(trackingId, lengthM, widthM);
            if (lastResponse.statusCode() != 503) {
                return lastResponse;
            }

            if (i < attempts - 1) {
                sleep(properties.getGateMatchRetryDelayMs());
            }
        }

        return lastResponse;
    }

    private AiHttpResponse sendGet(String path) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(buildUri(path))
                .timeout(Duration.ofMillis(properties.getHttpTimeoutMs()))
                .GET()
                .build();

        return execute(request);
    }

    private AiHttpResponse sendGateMatch(int trackingId, double lengthM, double widthM) {
        String payload = "{\"tracking_id\":" + trackingId +
                ",\"vehicle\":{\"length_m\":" + lengthM +
                ",\"width_m\":" + widthM + "}}";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(buildUri("/api/v1/gate/match"))
                .timeout(Duration.ofMillis(properties.getHttpTimeoutMs()))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload))
                .build();

        return execute(request);
    }

    private AiHttpResponse execute(HttpRequest request) {
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            JsonNode body = parseBody(response.body());
            return new AiHttpResponse(response.statusCode(), body);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            logger.error("AI request interrupted: {}", ex.getMessage());
            return new AiHttpResponse(503, objectMapper.createObjectNode().put("message", "AI service unreachable"));
        } catch (IOException ex) {
            logger.error("AI request failed: {}", ex.getMessage());
            return new AiHttpResponse(503, objectMapper.createObjectNode().put("message", "AI service unreachable"));
        }
    }

    private JsonNode parseBody(String body) {
        if (body == null || body.isBlank()) {
            return objectMapper.createObjectNode();
        }
        try {
            return objectMapper.readTree(body);
        } catch (IOException ex) {
            logger.warn("Could not parse AI response body as JSON");
            return objectMapper.createObjectNode().put("raw", body);
        }
    }

    private URI buildUri(String path) {
        String base = properties.getBaseUrl().endsWith("/")
                ? properties.getBaseUrl().substring(0, properties.getBaseUrl().length() - 1)
                : properties.getBaseUrl();
        return URI.create(base + path);
    }

    private void sleep(long delayMs) {
        try {
            Thread.sleep(delayMs);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    public record AiHttpResponse(int statusCode, JsonNode body) {
    }
}


