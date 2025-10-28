
Shader "Custom/GeneralizedKuwahara"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Radius ("Radius", Int) = 6
        _Sectors ("Sectors", Int) = 8
        _SamplesPerSector ("Samples Per Sector", Int) = 6
        _Q ("Q", Float) = 8.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalRenderPipeline" }

        Pass
        {
            Name "Generalized Kuwahara"
            Tags { "LightMode"="UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float2 uv           : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS   : SV_POSITION;
                float2 uv           : TEXCOORD0;
            };

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            CBUFFER_START(UnityPerMaterial)
                int _Radius;
                int _Sectors;
                int _SamplesPerSector;
                float _Q;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = input.uv;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                float2 uv = input.uv;
                float3 final_color = float3(0.0, 0.0, 0.0);
                float total_weight = 0.0;

                float sector_angle = 2.0 * 3.14159265359 / float(_Sectors);

                float3 means[8];
                float sigmas[8];

                for (int i = 0; i < _Sectors; i++)
                {
                    float theta_start = float(i) * sector_angle;
                    float theta_end = (float(i) + 1.0) * sector_angle;

                    float3 mean = float3(0.0, 0.0, 0.0);
                    float3 mean2 = float3(0.0, 0.0, 0.0);
                    int n = 0;

                    for (int r = 1; r <= _Radius; r++)
                    {
                        for (int s = 0; s < _SamplesPerSector; s++)
                        {
                            float t = (float(s) + 0.5) / float(_SamplesPerSector);
                            float ang = lerp(theta_start, theta_end, t);
                            float2 d;
                            d.x = float(r) * cos(ang);
                            d.y = float(r) * sin(ang);

                            float2 sample_uv = uv + d / _ScreenParams.xy;
                            float3 c = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, sample_uv).rgb;
                            mean += c;
                            mean2 += c * c;
                            n++;
                        }
                    }

                    mean /= float(n);
                    mean2 /= float(n);
                    float3 var = abs(mean2 - mean * mean);

                    means[i] = mean;
                    sigmas[i] = (var.r + var.g + var.b) / 3.0;
                }

                float sigma_sum = 0.0;
                for(int i = 0; i < _Sectors; i++)
                {
                    sigma_sum += sigmas[i];
                }

                for (int i = 0; i < _Sectors; i++)
                {
                    float w = exp(-_Q * sigmas[i] / (sigma_sum / _Sectors));
                    final_color += means[i] * w;
                    total_weight += w;
                }

                final_color /= total_weight;

                return half4(final_color, 1.0);
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
