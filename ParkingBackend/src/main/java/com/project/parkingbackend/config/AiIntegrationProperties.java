package com.project.parkingbackend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "ai")
public class AiIntegrationProperties {

    private String baseUrl = "http://127.0.0.1:8000";
    private String websocketUrl = "ws://127.0.0.1:8000/ws/live";

    private long mapPollIntervalMs = 1000;
    private int gateMatchMaxRetries = 5;
    private long gateMatchRetryDelayMs = 1200;

    private long wsReconnectBaseDelayMs = 1000;
    private long wsReconnectMaxDelayMs = 15000;

    private int httpTimeoutMs = 6000;

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String getWebsocketUrl() {
        return websocketUrl;
    }

    public void setWebsocketUrl(String websocketUrl) {
        this.websocketUrl = websocketUrl;
    }

    public long getMapPollIntervalMs() {
        return mapPollIntervalMs;
    }

    public void setMapPollIntervalMs(long mapPollIntervalMs) {
        this.mapPollIntervalMs = mapPollIntervalMs;
    }

    public int getGateMatchMaxRetries() {
        return gateMatchMaxRetries;
    }

    public void setGateMatchMaxRetries(int gateMatchMaxRetries) {
        this.gateMatchMaxRetries = gateMatchMaxRetries;
    }

    public long getGateMatchRetryDelayMs() {
        return gateMatchRetryDelayMs;
    }

    public void setGateMatchRetryDelayMs(long gateMatchRetryDelayMs) {
        this.gateMatchRetryDelayMs = gateMatchRetryDelayMs;
    }

    public long getWsReconnectBaseDelayMs() {
        return wsReconnectBaseDelayMs;
    }

    public void setWsReconnectBaseDelayMs(long wsReconnectBaseDelayMs) {
        this.wsReconnectBaseDelayMs = wsReconnectBaseDelayMs;
    }

    public long getWsReconnectMaxDelayMs() {
        return wsReconnectMaxDelayMs;
    }

    public void setWsReconnectMaxDelayMs(long wsReconnectMaxDelayMs) {
        this.wsReconnectMaxDelayMs = wsReconnectMaxDelayMs;
    }

    public int getHttpTimeoutMs() {
        return httpTimeoutMs;
    }

    public void setHttpTimeoutMs(int httpTimeoutMs) {
        this.httpTimeoutMs = httpTimeoutMs;
    }
}


