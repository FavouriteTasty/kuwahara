Shader "Unlit/GeneralizedKuwaharaFilter"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Radius ("Filter Size", Range(1, 25)) = 5
        _Sectors ("Sectors", Int) = 8
        _SamplesPerSector ("Samples Per Sector", Int) = 6
        _Q ("Q", Float) = 8.0
    }

    SubShader
    {
        Cull Off
        ZWrite On
        ZTest Always

        Tags { "RenderType" = "Opaque" "Renderpipeline" = "UniversalPipeline" }
    
        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            #if SHADER_API_GLES
                struct FilterAttributes
                {
                    float4 vertex : POSITION;
                    float2 uv : TEXCOORD0;
                };
            #else
                struct FilterAttributes
                {
                    uint vertex : SV_VertexID;
                };
            #endif
        
            struct FilterVaryings
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            int _Radius;
            int _Sectors;
            int _SamplesPerSector;
            float _Q;
            float4 _BlitTexture_TexelSize;

            FilterVaryings vert(FilterAttributes v)
            {
                FilterVaryings OUT;
                #if SHADER_API_GLES
                    float4 pos = input.vertex;
                    float2 uv = input.uv;
                #else
                    float4 pos = GetFullScreenTriangleVertexPosition(v.vertex);
                    float2 uv = GetFullScreenTriangleTexCoord(v.vertex);
                #endif

                OUT.vertex = pos;
                OUT.uv = uv;
                return OUT;
            }

            float4 frag(FilterVaryings i) : SV_TARGET
            {
                float3 final_color = float3(0.0, 0.0, 0.0);
                float total_weight = 0.0;
                float sector_angle = 2.0 * 3.14159265359 / float(_Sectors);

                float3 means[8];
                float sigmas[8];

                for (int index = 0; index < _Sectors; index++)
                {
                    float theta_start = float(index) * sector_angle;
                    float theta_end = (float(index) + 1.0) * sector_angle;

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

                            float2 sample_uv = i.uv + d / _ScreenParams.xy;
                            float3 c = SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearRepeat, sample_uv);

                            mean += c;
                            mean2 += c * c;
                            n++;
                        }
                    }

                    mean /= float(n);
                    mean2 /= float(n);
                    float3 var = abs(mean2 - mean * mean);

                    means[index] = mean;
                    sigmas[index] = (var.r + var.g + var.b) / 3.0;
                }

                float sigma_sum = 0.0;
                for(int index = 0; index < _Sectors; index++)
                {
                    sigma_sum += sigmas[index];
                }

                for (int index = 0; index < _Sectors; index++)
                {
                    float w = exp(-_Q * sigmas[index] / (sigma_sum / _Sectors));
                    final_color += means[index] * w;
                    total_weight += w;
                }

                final_color /= total_weight;

                return half4(final_color, 1.0);
            }
            ENDHLSL
        }
    }
}