// NBA Line Movement Analyzer v5.0
// Analyzes betting line movement and reverse line movement (RLM) detection.
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

type LineMovement struct {
	GameID        string  `json:"game_id"`
	Market        string  `json:"market"`
	OpeningLine   float64 `json:"opening_line"`
	CurrentLine   float64 `json:"current_line"`
	Movement      float64 `json:"movement"`
	IsRLM         bool    `json:"is_rlm"`
	SharpSide     string  `json:"sharp_side"`
	PublicPercent float64 `json:"public_percent"`
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
		json.NewEncoder(w).Encode(Health{Service: "line-movement-analyzer", Status: "ok"})
	})

	// Analyze line movement for a game
	mux.HandleFunc("/analyze", func(w http.ResponseWriter, r *http.Request) {
		gameID := r.URL.Query().Get("game_id")
		if gameID == "" {
			http.Error(w, "missing game_id", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		movement := LineMovement{
			GameID:        gameID,
			Market:        "spread",
			OpeningLine:   -6.0,
			CurrentLine:   -6.5,
			Movement:      -0.5,
			IsRLM:         true,
			SharpSide:     "home",
			PublicPercent: 45.0,
		}
		json.NewEncoder(w).Encode(movement)
	})

	addr := getenv("LISTEN_ADDR", ":8084")
	log.Printf("NBA Line Movement Analyzer listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
