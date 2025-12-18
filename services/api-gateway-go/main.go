// NBA Prediction API Gateway v5.0
// Minimal HTTP gateway that proxies requests to backend services.
//
// NOTE: This is a scaffolded gateway. The primary API is strict-api (port 8090).
package main

import (
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

// Service URLs (override via env)
var (
	predictionURL = getenv("PREDICTION_URL", "http://localhost:8082")
	featureURL    = getenv("FEATURE_URL", "http://localhost:8081")
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	mux := http.NewServeMux()

	// Health
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"service":"api-gateway","status":"ok","note":"Use strict-api:8090 for predictions"}`))
	})

	// Predict proxy -> prediction service
	mux.HandleFunc("/api/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		client := &http.Client{Timeout: 15 * time.Second}
		req, err := http.NewRequest(http.MethodPost, predictionURL+"/predict", r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
		resp, err := client.Do(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		for k, vv := range resp.Header {
			for _, v := range vv {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	})

	// Features proxy -> feature store
	mux.HandleFunc("/api/features", func(w http.ResponseWriter, r *http.Request) {
		client := &http.Client{Timeout: 10 * time.Second}
		req, err := http.NewRequest(http.MethodGet, featureURL+"/features?"+r.URL.RawQuery, nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		resp, err := client.Do(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		for k, vv := range resp.Header {
			for _, v := range vv {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	})

	// Odds endpoint - redirect to strict-api
	// NOTE: odds-ingestion is a background worker, not an HTTP service
	// Live odds are fetched directly by strict-api via The Odds API
	mux.HandleFunc("/api/odds", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"Odds not available via gateway. Use strict-api:8090/slate/{date} instead."}`))
	})

	addr := getenv("LISTEN_ADDR", ":8080")
	log.Printf("NBA API Gateway listening on %s (prediction=%s, features=%s)", addr, predictionURL, featureURL)
	log.Printf("NOTE: For full predictions, use strict-api at port 8090")
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
