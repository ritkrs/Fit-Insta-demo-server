import requests

# Meta App + Instagram Variables
ig_user_id = "17841472117168408"
app_id = "629232336425571"
app_secret = "e18fff02092b87e138b6528ccfa4a1ce"
user_access_token = "EAAHxwjnB7HABOzlRZCo5IZCD8YR5h2At8A0MkOhUnWcwuO71S5mv12Jq1kztpDE462TBhRJmuIk7m4DflPDxRTi0TsNHZCZBuZBPPh6TgojaPrEU7Bpi15ZB6SssYuZBruP9R9MrZCTkVpwy3cyZCjGFsWZA2L5UdZCNcDNTMCUA0jANOZAoihjMbVzoMaC7IkfE5ZA7TWAvnACMRKQJ2WDMKNMtK3odBcgZDZD"

# Get Long Access Token
url = f"https://graph.facebook.com/v17.0/oauth/access_token?grant_type=fb_exchange_token&client_id={app_id}&client_secret={app_secret}&fb_exchange_token={user_access_token}"

response = requests.get(url)

# Extract long-lived access token
long_access_token = response.json()

print(long_access_token)
