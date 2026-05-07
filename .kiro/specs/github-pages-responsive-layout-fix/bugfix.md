# Bugfix Requirements Document

## Introduction

The GitHub Pages site has a responsive layout issue at viewport widths near the 960px breakpoint. When users resize their browser window to approximately half-width on a 1920px display (~960px viewport width), the layout appears cramped and poorly adapted. The site uses a fixed sidebar width (340px) with fixed content margins (320px), which creates a narrow content area at medium screen sizes. The responsive breakpoint at 960px switches abruptly from desktop to mobile layout without smooth adaptation in the transition zone.

This bugfix addresses the layout problems at viewport widths between 961px and 1200px, where the desktop layout remains active but becomes increasingly cramped, and ensures smooth responsive behavior across all screen sizes.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN viewport width is between 961px and 1200px THEN the system maintains desktop layout with fixed sidebar (340px) and fixed content margin (320px), creating a cramped content area of only 640px-880px width

1.2 WHEN viewport width is exactly at or near 960px THEN the system creates an abrupt transition between desktop and mobile layouts with no intermediate responsive adaptation

1.3 WHEN viewport width is between 961px and 1200px THEN the system displays content with `margin-left: 320px` and `width: calc(100% - 320px)`, leaving insufficient space for readable content layout

1.4 WHEN viewport width is at the 960px breakpoint boundary THEN the system switches layout modes abruptly without smooth visual transition or intermediate states

### Expected Behavior (Correct)

2.1 WHEN viewport width is between 961px and 1200px THEN the system SHALL adjust the sidebar and content widths proportionally to maintain readable content area (minimum 600px content width)

2.2 WHEN viewport width transitions across the 960px breakpoint THEN the system SHALL provide smooth visual adaptation without abrupt layout shifts

2.3 WHEN viewport width is between 961px and 1200px THEN the system SHALL reduce sidebar width or adjust content margins to ensure adequate content display area

2.4 WHEN viewport width is at any size between 320px and 1920px THEN the system SHALL display content without horizontal scrolling, content cutoff, or cramped layout

### Unchanged Behavior (Regression Prevention)

3.1 WHEN viewport width is greater than 1200px THEN the system SHALL CONTINUE TO display the full desktop layout with 340px sidebar and proper content spacing

3.2 WHEN viewport width is 960px or less THEN the system SHALL CONTINUE TO display the mobile layout with full-width stacked content and collapsible navigation

3.3 WHEN users interact with navigation elements THEN the system SHALL CONTINUE TO provide the same navigation functionality and link behavior

3.4 WHEN users view content on mobile devices (≤480px) THEN the system SHALL CONTINUE TO display the optimized mobile layout with touch-friendly controls

3.5 WHEN users print pages THEN the system SHALL CONTINUE TO apply print-specific styles for optimal printed output

3.6 WHEN users have reduced motion preferences THEN the system SHALL CONTINUE TO respect accessibility settings and disable animations

3.7 WHEN users navigate with keyboard THEN the system SHALL CONTINUE TO provide visible focus indicators and accessible navigation
