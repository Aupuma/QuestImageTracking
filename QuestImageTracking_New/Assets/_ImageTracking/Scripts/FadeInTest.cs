using System.Collections;
using UnityEngine;

public class FadeInTest : MonoBehaviour
{
    Color endColor;
    Vector3 originPosition;
    Vector3 endPosition;
    Renderer renderer;

    void Awake()
    {
        renderer = GetComponent<Renderer>();
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        endColor = renderer.material.color;
        endPosition = transform.localPosition;
        originPosition = transform.localPosition;
        originPosition.z = -0.4f;
    }

    // Update is called once per frame
    public void FadeIn()
    {
        // Start the fade-in coroutine
        StartCoroutine(FadeInCoroutine(0.25f));
    }

    private IEnumerator FadeInCoroutine(float duration)
    {
        Color color = endColor;

        float startTime = Time.time;
        while (Time.time < startTime + duration)
        {
            color.a = Mathf.Lerp(0f, 1f, (Time.time - startTime) / duration);
            renderer.material.color = color;
            transform.localPosition = Vector3.Lerp(originPosition, endPosition, (Time.time - startTime) / duration);
            yield return null;
        }

        color.a = 1f;
        renderer.material.color = color;
        transform.localPosition = endPosition;
    }
}
