// Shared interaction for the named-resource manager. Use a dedicated test library.
export async function saveThroughLibrary(page) {
  await page.locator('#save').click();
  await page.locator('#library').waitFor({state:'visible'});
  if (await page.locator('#library-folder').inputValue() !== 'user') {
    await page.locator('#library-folder').selectOption('user');
  }
  await page.waitForFunction(()=>!document.querySelector('#resource-save').disabled);
  await page.locator('#resource-save').click();
  await page.waitForFunction(()=>document.querySelector('#library-message').textContent.startsWith('已保存'));
  await page.locator('#library-close').click();
}
