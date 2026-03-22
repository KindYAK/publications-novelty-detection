$dir = "C:\Users\HaveToCook\novelty-search\data\processed\v3"
Get-ChildItem "$dir\arxiv_ids_*" | Sort-Object Name | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host ("{0,6} MB  {1}" -f $sizeMB, $_.Name)
}
