import requests

def count_events():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80"
    response = requests.get(url)
    data = response.json()
    events = data["events"]
    count = len(events)
    print(f"Number of events: {count}")

    for event in events:
        print(event["name"] + " - " + event["date"])

count_events()