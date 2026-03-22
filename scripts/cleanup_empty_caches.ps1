$dir = "C:\Users\HaveToCook\novelty-search\data\processed\v3"
Get-ChildItem "$dir\*.json" | Where-Object { $_.Length -le 2 } | ForEach-Object {
    Remove-Item $_.FullName
    Write-Host "Deleted: $($_.Name)"
}
Write-Host "Done."
