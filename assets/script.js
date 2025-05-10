var stopDownloadButtonTarget = null;

function getTargetForStopDownload(url){
    return stopDownloadButtonTarget;
}

function stopDownload(url){
    stopDownloadButtonTarget = url;

    document.querySelector('#stop_download').click()
}

function setupDownloadUpdates(){
    button = document.querySelector('#refresh_downloads');
    if(button && document.querySelector('.downloads .download.active'))
        button.click();

    setTimeout(setupDownloadUpdates, 1000);
}

setupDownloadUpdates();