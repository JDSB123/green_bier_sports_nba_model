// NBA Feature Store v5.0
// High-performance feature serving for NBA predictions.
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
)

type Health struct {
	Service string `json:"service"`
	Status  string `json:"status"`
}

type FeatureResponse struct {
	Team   string             `json:"team"`
	Date   string             `json:"date"`
	Values map[string]float64 `json:"values"`
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(Health{Service: "feature-store", Status: "ok"})
	})

	// Feature endpoint: returns computed features for a team
	mux.HandleFunc("/features", func(w http.ResponseWriter, r *http.Request) {
		team := r.URL.Query().Get("team")
		date := r.URL.Query().Get("date")
		if team == "" {
			http.Error(w, "missing team", http.StatusBadRequest)
			return
		}
		if date == "" {
			date = "today"
		}

		w.Header().Set("Content-Type", "application/json")
		resp := FeatureResponse{
			Team: team,
			Date: date,
			Values: map[string]float64{
				"recent_form":          0.12,
				"travel_penalty":       0.0,
				"injury_adj":           0.0,
				"rest_advantage":       0.0,
				"home_court_advantage": 3.5,
			},
		}
		json.NewEncoder(w).Encode(resp)
	})

	addr := getenv("LISTEN_ADDR", ":8081")
	log.Printf("NBA Feature Store listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
