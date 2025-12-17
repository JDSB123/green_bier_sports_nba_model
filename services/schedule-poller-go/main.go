// NBA Schedule Poller v5.0
// Polls game schedules from API-Basketball and other sources.
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

type Game struct {
	GameID       string `json:"game_id"`
	HomeTeam     string `json:"home_team"`
	AwayTeam     string `json:"away_team"`
	CommenceTime string `json:"commence_time"`
	Status       string `json:"status"`
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
		json.NewEncoder(w).Encode(Health{Service: "schedule-poller", Status: "ok"})
	})

	// Get games for a date
	mux.HandleFunc("/games", func(w http.ResponseWriter, r *http.Request) {
		date := r.URL.Query().Get("date")
		if date == "" {
			date = "today"
		}

		w.Header().Set("Content-Type", "application/json")
		games := []Game{
			{
				GameID:       "game-1",
				HomeTeam:     "Lakers",
				AwayTeam:     "Celtics",
				CommenceTime: "2025-12-18T19:30:00Z",
				Status:       "scheduled",
			},
		}
		json.NewEncoder(w).Encode(games)
	})

	addr := getenv("LISTEN_ADDR", ":8085")
	log.Printf("NBA Schedule Poller listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
